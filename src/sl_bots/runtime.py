"""CPU 推理运行时、SPSC 队列和逐 Bot 回退规则。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import ctypes
from ctypes import wintypes
import os
import math
from pathlib import Path
import random
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contracts import (
    MAX_BOTS,
    ActionBatchV1,
    BotActionV1,
    BotObservationV1,
    ObservationBatchV1,
)


ACTION_MASK_ALL = 0xFF


def _action_from_onnx_outputs(
    action_vector: Sequence[float],
    button_logits: Sequence[float],
    weapon_logits: Sequence[float],
    buy_logits: Sequence[float],
    *,
    server_tick: int,
    movement_alpha: Sequence[float] | None = None,
    movement_beta: Sequence[float] | None = None,
    mouse_loc: Sequence[float] | None = None,
    mouse_scale: Sequence[float] | None = None,
    mouse_mix_logits: Sequence[Sequence[float]] | None = None,
    random_seed: int | None = None,
) -> BotActionV1:
    if len(action_vector) < 5:
        raise ValueError("ONNX action output must contain five continuous values")
    continuous = [float(value) for value in action_vector[:5]]
    if not all(math.isfinite(value) for value in continuous):
        raise ValueError("ONNX continuous action output must be finite")
    if len(button_logits) != 32 or len(weapon_logits) == 0 or len(buy_logits) == 0:
        raise ValueError("ONNX action heads have invalid dimensions")
    if not all(math.isfinite(float(value)) for value in button_logits):
        raise ValueError("ONNX button action output must be finite")
    if not all(math.isfinite(float(value)) for value in weapon_logits) or not all(
        math.isfinite(float(value)) for value in buy_logits
    ):
        raise ValueError("ONNX categorical action output must be finite")
    distribution_sampling = movement_alpha is not None or movement_beta is not None
    if distribution_sampling:
        if movement_alpha is None or movement_beta is None:
            raise ValueError("ONNX movement distribution outputs must be provided together")
        if len(movement_alpha) != 3 or len(movement_beta) != 3:
            raise ValueError("ONNX movement distribution outputs have invalid dimensions")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (*movement_alpha, *movement_beta)
        ):
            raise ValueError("ONNX movement distribution outputs must be finite and positive")
    rng = random.Random(server_tick if random_seed is None else random_seed)
    mouse_values = continuous[3:5]
    if any(value is not None for value in (mouse_loc, mouse_scale, mouse_mix_logits)):
        if mouse_loc is None or mouse_scale is None or mouse_mix_logits is None:
            raise ValueError("ONNX mouse distribution outputs must be provided together")
        if len(mouse_loc) != 2 or len(mouse_scale) != 2 or len(mouse_mix_logits) != 2:
            raise ValueError("ONNX mouse distribution outputs have invalid dimensions")
        if any(len(row) != 3 for row in mouse_mix_logits):
            raise ValueError("ONNX mouse mixture logits must contain three components per axis")
        if not all(math.isfinite(float(value)) for value in mouse_loc) or not all(
            math.isfinite(float(value)) and float(value) > 0.0 for value in mouse_scale
        ) or not all(math.isfinite(float(value)) for row in mouse_mix_logits for value in row):
            raise ValueError("ONNX mouse distribution outputs must be finite")
        sampled = []
        for axis in range(2):
            logits = [float(value) for value in mouse_mix_logits[axis]]
            maximum = max(logits)
            weights = [math.exp(value - maximum) for value in logits]
            total = sum(weights)
            threshold = rng.random() * total
            component = 0
            cumulative = 0.0
            for component, weight in enumerate(weights):
                cumulative += weight
                if threshold <= cumulative:
                    break
            probability = min(1.0 - 1e-6, max(1e-6, rng.random()))
            logistic_noise = math.log(probability / (1.0 - probability))
            offset = (-1.0, 0.0, 1.0)[component]
            sampled.append(float(mouse_loc[axis]) + (offset + logistic_noise) * float(mouse_scale[axis]))
        mouse_values = sampled
    if distribution_sampling:
        continuous[:3] = [
            2.0 * rng.betavariate(float(alpha), float(beta)) - 1.0
            for alpha, beta in zip(movement_alpha, movement_beta)
        ]
    continuous[3:5] = mouse_values
    if distribution_sampling:
        buttons = sum(
            1 << index
            for index, value in enumerate(button_logits)
            if rng.random() < (
                1.0 / (1.0 + math.exp(-float(value)))
                if float(value) >= 0.0
                else math.exp(float(value)) / (1.0 + math.exp(float(value)))
            )
        )

        def sample_categorical(logits: Sequence[float]) -> int:
            maximum = max(float(value) for value in logits)
            weights = [math.exp(float(value) - maximum) for value in logits]
            threshold = rng.random() * sum(weights)
            cumulative = 0.0
            for index, weight in enumerate(weights):
                cumulative += weight
                if threshold <= cumulative:
                    return index
            return len(weights) - 1

        weapon_index = sample_categorical(weapon_logits)
        buy_index = sample_categorical(buy_logits)
    else:
        buttons = sum(
            1 << index
            for index, value in enumerate(button_logits)
            if float(value) >= 0.0
        )
        weapon_index = max(range(len(weapon_logits)), key=lambda index: float(weapon_logits[index]))
        buy_index = max(range(len(buy_logits)), key=lambda index: float(buy_logits[index]))
    return BotActionV1(
        target_tick=int(server_tick) + 1,
        forward=max(-1.0, min(1.0, continuous[0])),
        side=max(-1.0, min(1.0, continuous[1])),
        up=max(-1.0, min(1.0, continuous[2])),
        yaw_delta_deg=max(-180.0, min(180.0, continuous[3])),
        pitch_delta_deg=max(-90.0, min(90.0, continuous[4])),
        buttons=buttons,
        weapon_select=-1 if weapon_index == 0 else weapon_index,
        buy_action=buy_index,
        action_valid_mask=ACTION_MASK_ALL,
    )


class QueueFull(RuntimeError):
    """Raised when a producer would overtake the SPSC consumer."""


class QueueAborted(RuntimeError):
    """Raised after the shared-memory owner has aborted the queue."""


class QueueEmpty(RuntimeError):
    """Raised only by the explicit raising queue API."""


class SharedMemoryRingV1:
    """线程安全的 SPSC 语义模型；原生扩展使用同样的序号规则。"""

    def __init__(self, capacity: int = 64) -> None:
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._slots: list[bytes | None] = [None] * capacity
        self._write_sequence = 0
        self._read_sequence = 0
        self._aborted = False
        self._lock = threading.Lock()
        self._data_event = threading.Event()

    @property
    def size(self) -> int:
        with self._lock:
            return self._write_sequence - self._read_sequence

    @property
    def aborted(self) -> bool:
        return self._aborted

    def publish(self, packet: bytes) -> int:
        packet = bytes(packet)
        with self._lock:
            if self._aborted:
                raise QueueAborted("queue is aborted")
            if self._write_sequence - self._read_sequence >= self.capacity:
                raise QueueFull("SPSC queue is full")
            sequence = self._write_sequence
            self._slots[sequence % self.capacity] = packet
            self._write_sequence += 1
            self._data_event.set()
            return sequence

    def try_read(self) -> bytes | None:
        with self._lock:
            if self._aborted:
                raise QueueAborted("queue is aborted")
            if self._read_sequence >= self._write_sequence:
                return None
            sequence = self._read_sequence
            packet = self._slots[sequence % self.capacity]
            self._slots[sequence % self.capacity] = None
            self._read_sequence += 1
            if self._read_sequence >= self._write_sequence:
                self._data_event.clear()
            if packet is None:
                raise RuntimeError("SPSC queue slot was published without a packet")
            return packet

    def read(self) -> bytes:
        packet = self.try_read()
        if packet is None:
            raise QueueEmpty("SPSC queue is empty")
        return packet

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            self._slots = [None] * self.capacity
            self._data_event.set()

    def reset(self) -> None:
        with self._lock:
            self._slots = [None] * self.capacity
            self._write_sequence = 0
            self._read_sequence = 0
            self._aborted = False
            self._data_event.clear()

    def wait_for_data(self, timeout_ms: int) -> bool:
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
            raise ValueError("timeout_ms must be a non-negative integer")
        if self.aborted:
            return False
        if self.size > 0:
            return True
        self._data_event.wait(timeout_ms / 1000.0)
        return not self.aborted and self.size > 0


class SharedMemoryTransportV1:
    """两条固定方向队列的协议适配器。"""

    def __init__(self, *, capacity: int = 64, epoch: int = 0) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        self.observations = SharedMemoryRingV1(capacity)
        self.actions = SharedMemoryRingV1(capacity)
        from .get5_control import CONTROL_EVENT_SIZE

        self.control = SharedMemoryRingV1(capacity)
        self.control_packet_size = CONTROL_EVENT_SIZE
        self._heartbeat = time.monotonic_ns()

    def _check_epoch(self, epoch: int) -> None:
        if int(epoch) != self.epoch:
            raise ValueError(f"epoch mismatch: expected {self.epoch}, got {epoch}")

    def publish_observation(self, batch: ObservationBatchV1) -> int:
        self._check_epoch(batch.epoch)
        sequence = self.observations.publish(batch.pack())
        self._heartbeat = time.monotonic_ns()
        return sequence

    def try_read_observation(self) -> ObservationBatchV1 | None:
        packet = self.observations.try_read()
        if packet is None:
            return None
        batch = ObservationBatchV1.unpack(packet)
        self._check_epoch(batch.epoch)
        self._heartbeat = time.monotonic_ns()
        return batch

    def publish_action(self, batch: ActionBatchV1) -> int:
        self._check_epoch(batch.epoch)
        sequence = self.actions.publish(batch.pack())
        self._heartbeat = time.monotonic_ns()
        return sequence

    def try_read_action(self) -> ActionBatchV1 | None:
        packet = self.actions.try_read()
        if packet is None:
            return None
        batch = ActionBatchV1.unpack(packet)
        self._check_epoch(batch.epoch)
        self._heartbeat = time.monotonic_ns()
        return batch

    def wait_for_observation(self, timeout_ms: int) -> bool:
        return self.observations.wait_for_data(timeout_ms)

    def publish_control(self, event: Any) -> int:
        from .get5_control import Get5ControlEventV1

        payload = event.pack() if isinstance(event, Get5ControlEventV1) else bytes(event)
        if len(payload) != self.control_packet_size:
            raise ValueError(f"control event must be {self.control_packet_size} bytes")
        sequence = self.control.publish(payload)
        self._heartbeat = time.monotonic_ns()
        return sequence

    def try_read_control(self) -> Any | None:
        from .get5_control import Get5ControlEventV1

        packet = self.control.try_read()
        if packet is None:
            return None
        self._heartbeat = time.monotonic_ns()
        return Get5ControlEventV1.unpack(packet)

    def get_heartbeat(self) -> int:
        return self._heartbeat

    def refresh_epoch(self) -> int:
        return self.epoch

    def abort(self) -> None:
        self.observations.abort()
        self.actions.abort()
        self.control.abort()

    def switch_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        self.observations.reset()
        self.actions.reset()
        self.control.reset()
        self._heartbeat = time.monotonic_ns()


SharedMemoryRing = SharedMemoryRingV1
SharedMemoryTransport = SharedMemoryTransportV1


class _Win32SharedMemoryRingV1:
    """与 native/ipc 的 Win32 文件映射环共享同一内存布局。"""

    _HEADER = struct.Struct("<IIIIqqiiq")
    _RING_MAGIC = 0x534C4252
    _RING_VERSION = 1
    _MAX_CAPACITY = 1024
    _PAGE_READWRITE = 0x04
    _FILE_MAP_ALL_ACCESS = 0x000F001F
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def __init__(
        self,
        name: str,
        *,
        epoch: int,
        capacity: int,
        packet_size: int,
        open_existing: bool = False,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("Win32 shared memory requires Windows")
        if not name or not isinstance(name, str):
            raise ValueError("mapping name must be a non-empty string")
        if not 1 <= capacity <= self._MAX_CAPACITY:
            raise ValueError(f"capacity must be between 1 and {self._MAX_CAPACITY}")
        if not 1 <= packet_size <= 0xFFFFFFFF:
            raise ValueError("packet_size must fit in uint32")
        if not 0 <= epoch < 1 << 32:
            raise ValueError("epoch must fit in uint32")
        self.name = name
        self.capacity = capacity
        self.packet_size = packet_size
        self.open_existing = bool(open_existing)
        self.mapping_size = self._HEADER.size + capacity * packet_size
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_api()
        self._mapping = self._create_mapping()
        self._address = self._map_view()
        self._initialize_or_validate(epoch)

    def _configure_api(self) -> None:
        self._kernel32.CreateFileMappingW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPCWSTR,
        ]
        self._kernel32.CreateFileMappingW.restype = wintypes.HANDLE
        self._kernel32.OpenFileMappingW.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._kernel32.OpenFileMappingW.restype = wintypes.HANDLE
        self._kernel32.MapViewOfFile.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_size_t,
        ]
        self._kernel32.MapViewOfFile.restype = wintypes.LPVOID
        self._kernel32.UnmapViewOfFile.argtypes = [wintypes.LPCVOID]
        self._kernel32.UnmapViewOfFile.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.GetTickCount64.argtypes = []
        self._kernel32.GetTickCount64.restype = ctypes.c_ulonglong
        self._sync = ctypes.WinDLL("api-ms-win-core-synch-l1-2-0.dll", use_last_error=True)
        self._sync.WaitOnAddress.argtypes = [
            wintypes.LPVOID,
            wintypes.LPCVOID,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        self._sync.WaitOnAddress.restype = wintypes.BOOL

    def _create_mapping(self) -> Any:
        if self.open_existing:
            mapping = self._kernel32.OpenFileMappingW(
                self._FILE_MAP_ALL_ACCESS,
                False,
                self.name,
            )
            if not mapping:
                error = ctypes.get_last_error()
                raise OSError(error, f"OpenFileMappingW failed for {self.name}")
            self._created = False
            return mapping
        high = (self.mapping_size >> 32) & 0xFFFFFFFF
        low = self.mapping_size & 0xFFFFFFFF
        mapping = self._kernel32.CreateFileMappingW(
            self._INVALID_HANDLE_VALUE,
            None,
            self._PAGE_READWRITE,
            high,
            low,
            self.name,
        )
        if not mapping:
            error = ctypes.get_last_error()
            raise OSError(error, f"CreateFileMappingW failed for {self.name}")
        self._created = ctypes.get_last_error() != 183
        return mapping

    def _map_view(self) -> int:
        address = self._kernel32.MapViewOfFile(
            self._mapping,
            self._FILE_MAP_ALL_ACCESS,
            0,
            0,
            self.mapping_size,
        )
        if not address:
            error = ctypes.get_last_error()
            self._kernel32.CloseHandle(self._mapping)
            self._mapping = None
            raise OSError(error, f"MapViewOfFile failed for {self.name}")
        return int(address)

    def _header(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        return self._HEADER.unpack(ctypes.string_at(self._address, self._HEADER.size))

    def _initialize_or_validate(self, epoch: int) -> None:
        magic, version, capacity, packet_size, _, _, stored_epoch, aborted, _ = self._header()
        if self._created:
            ctypes.memset(self._address, 0, self.mapping_size)
            self._write_header_fields(
                magic=self._RING_MAGIC,
                version=self._RING_VERSION,
                capacity=self.capacity,
                packet_size=self.packet_size,
                epoch=epoch,
                aborted=0,
            )
            self._write_counter(40, self._now())
            return
        if (
            magic != self._RING_MAGIC
            or version != self._RING_VERSION
            or capacity != self.capacity
            or packet_size != self.packet_size
            or (stored_epoch & 0xFFFFFFFF) != epoch
            or aborted not in (0, 1)
        ):
            self.close()
            raise ValueError(f"shared memory ring header mismatch for {self.name}")

    def _write_header_fields(
        self,
        *,
        magic: int,
        version: int,
        capacity: int,
        packet_size: int,
        epoch: int,
        aborted: int,
    ) -> None:
        current = list(self._header())
        current[0:4] = [magic, version, capacity, packet_size]
        current[6] = ctypes.c_int32(epoch).value
        current[7] = aborted
        ctypes.memmove(self._address, self._HEADER.pack(*current), self._HEADER.size)

    def _read_counter(self, offset: int) -> int:
        return int(ctypes.c_longlong.from_address(self._address + offset).value)

    def _write_counter(self, offset: int, value: int) -> None:
        ctypes.c_longlong.from_address(self._address + offset).value = value

    def _read_aborted(self) -> int:
        return int(ctypes.c_int32.from_address(self._address + 36).value)

    @property
    def aborted(self) -> bool:
        return self._read_aborted() != 0

    def current_epoch(self) -> int:
        return self._header()[6] & 0xFFFFFFFF

    def _now(self) -> int:
        return int(self._kernel32.GetTickCount64())

    def _touch(self) -> None:
        self._write_counter(40, self._now())

    def _slot_address(self, index: int) -> int:
        return self._address + self._HEADER.size + (index % self.capacity) * self.packet_size

    def publish(self, packet: bytes) -> int:
        packet = bytes(packet)
        if len(packet) != self.packet_size:
            raise ValueError(f"packet must be {self.packet_size} bytes")
        if self._read_aborted():
            raise QueueAborted("queue is aborted")
        write_index = self._read_counter(16)
        read_index = self._read_counter(24)
        if write_index - read_index >= self.capacity:
            raise QueueFull("SPSC queue is full")
        ctypes.memmove(self._slot_address(write_index), packet, self.packet_size)
        self._write_counter(16, write_index + 1)
        self._touch()
        return write_index

    def try_read(self) -> bytes | None:
        if self._read_aborted():
            raise QueueAborted("queue is aborted")
        write_index = self._read_counter(16)
        read_index = self._read_counter(24)
        if read_index >= write_index:
            return None
        packet = ctypes.string_at(self._slot_address(read_index), self.packet_size)
        self._write_counter(24, read_index + 1)
        self._touch()
        return packet

    def wait_for_data(self, timeout_ms: int) -> bool:
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
            raise ValueError("timeout_ms must be a non-negative integer")
        if self.aborted:
            return False
        write_index = self._read_counter(16)
        if self._read_counter(24) < write_index:
            return True
        compare = ctypes.c_longlong(write_index)
        self._sync.WaitOnAddress(
            ctypes.c_void_p(self._address + 16),
            ctypes.byref(compare),
            ctypes.sizeof(compare),
            timeout_ms,
        )
        return not self.aborted and self._read_counter(24) < self._read_counter(16)

    def abort(self) -> None:
        ctypes.c_int32.from_address(self._address + 36).value = 1
        self._touch()

    def switch_epoch(self, epoch: int) -> None:
        if not 0 <= epoch < 1 << 32:
            raise ValueError("epoch must fit in uint32")
        self._write_counter(16, 0)
        self._write_counter(24, 0)
        ctypes.c_int32.from_address(self._address + 32).value = ctypes.c_int32(epoch).value
        ctypes.c_int32.from_address(self._address + 36).value = 0
        self._touch()

    def heartbeat(self) -> int:
        return self._read_counter(40)

    def close(self) -> None:
        address = getattr(self, "_address", None)
        mapping = getattr(self, "_mapping", None)
        if address:
            self._kernel32.UnmapViewOfFile(ctypes.c_void_p(address))
            self._address = None
        if mapping:
            self._kernel32.CloseHandle(mapping)
            self._mapping = None

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


class Win32SharedMemoryTransportV1:
    """Windows named mappings consumed by the SourceMod bridge and CPU service."""

    def __init__(
        self,
        name: str,
        *,
        capacity: int = 64,
        epoch: int = 0,
        open_existing: bool = False,
    ) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.name = str(name)
        self.epoch = int(epoch)
        from .get5_control import CONTROL_EVENT_SIZE

        try:
            self.observations = _Win32SharedMemoryRingV1(
                f"{self.name}_obs_v1",
                epoch=self.epoch,
                capacity=capacity,
                packet_size=ObservationBatchV1(0, 0, ()).packet_size,
                open_existing=open_existing,
            )
            self.actions = _Win32SharedMemoryRingV1(
                f"{self.name}_act_v1",
                epoch=self.epoch,
                capacity=capacity,
                packet_size=ActionBatchV1(0, 0, ()).packet_size,
                open_existing=open_existing,
            )
            self.control = _Win32SharedMemoryRingV1(
                f"{self.name}_ctl_v1",
                epoch=self.epoch,
                capacity=capacity,
                packet_size=CONTROL_EVENT_SIZE,
                open_existing=open_existing,
            )
        except Exception:
            observations = getattr(self, "observations", None)
            actions = getattr(self, "actions", None)
            control = getattr(self, "control", None)
            if observations is not None:
                observations.close()
            if actions is not None:
                actions.close()
            if control is not None:
                control.close()
            raise

    def _check_epoch(self, epoch: int) -> None:
        if int(epoch) != self.epoch:
            raise ValueError(f"epoch mismatch: expected {self.epoch}, got {epoch}")

    def refresh_epoch(self) -> int:
        observation_epoch = self.observations.current_epoch()
        action_epoch = self.actions.current_epoch()
        control_epoch = self.control.current_epoch()
        if observation_epoch != action_epoch or observation_epoch != control_epoch:
            raise ValueError(
                "shared memory epoch mismatch: "
                f"observations={observation_epoch}, actions={action_epoch}, control={control_epoch}"
            )
        self.epoch = observation_epoch
        return self.epoch

    def publish_observation(self, batch: ObservationBatchV1) -> int:
        self.refresh_epoch()
        self._check_epoch(batch.epoch)
        sequence = self.observations.publish(batch.pack())
        return sequence

    def try_read_observation(self) -> ObservationBatchV1 | None:
        self.refresh_epoch()
        packet = self.observations.try_read()
        if packet is None:
            return None
        batch = ObservationBatchV1.unpack(packet)
        self._check_epoch(batch.epoch)
        return batch

    def publish_action(self, batch: ActionBatchV1) -> int:
        self.refresh_epoch()
        self._check_epoch(batch.epoch)
        return self.actions.publish(batch.pack())

    def try_read_action(self) -> ActionBatchV1 | None:
        self.refresh_epoch()
        packet = self.actions.try_read()
        if packet is None:
            return None
        batch = ActionBatchV1.unpack(packet)
        self._check_epoch(batch.epoch)
        return batch

    def wait_for_observation(self, timeout_ms: int) -> bool:
        self.refresh_epoch()
        return self.observations.wait_for_data(timeout_ms)

    def publish_control(self, event: Any) -> int:
        from .get5_control import Get5ControlEventV1, CONTROL_EVENT_SIZE

        payload = event.pack() if isinstance(event, Get5ControlEventV1) else bytes(event)
        if len(payload) != CONTROL_EVENT_SIZE:
            raise ValueError(f"control event must be {CONTROL_EVENT_SIZE} bytes")
        return self.control.publish(payload)

    def try_read_control(self) -> Any | None:
        from .get5_control import Get5ControlEventV1

        packet = self.control.try_read()
        if packet is None:
            return None
        return Get5ControlEventV1.unpack(packet)

    def get_heartbeat(self) -> int:
        return max(
            self.observations.heartbeat(),
            self.actions.heartbeat(),
            self.control.heartbeat(),
        )

    def abort(self) -> None:
        self.observations.abort()
        self.actions.abort()
        self.control.abort()

    def switch_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.observations.switch_epoch(int(epoch))
        self.actions.switch_epoch(int(epoch))
        self.control.switch_epoch(int(epoch))
        self.epoch = int(epoch)

    def close(self) -> None:
        self.observations.close()
        self.actions.close()
        self.control.close()

    def __enter__(self) -> "Win32SharedMemoryTransportV1":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass
class BotRuntimeStateV1:
    entity_id: int | None = None
    hidden: Any = None
    consecutive_missing: int = 0
    recent_missing: deque[bool] = field(default_factory=lambda: deque(maxlen=128))
    permanent_fallback: bool = False
    state_fault: bool = False

    @property
    def recent_missing_count(self) -> int:
        return sum(self.recent_missing)


def _observation_entity_id(observation: bytes) -> int | None:
    if len(observation) < 10:
        return None
    entity_id = struct.unpack_from("<H", observation, 8)[0]
    return entity_id if entity_id > 0 else None


def neutral_action(target_tick: int) -> BotActionV1:
    return BotActionV1(
        target_tick=target_tick,
        forward=0.0,
        side=0.0,
        up=0.0,
        yaw_delta_deg=0.0,
        pitch_delta_deg=0.0,
        buttons=0,
        weapon_select=-1,
        buy_action=0,
        action_valid_mask=ACTION_MASK_ALL,
    )


def valve_fallback_action(target_tick: int) -> BotActionV1:
    action = neutral_action(target_tick)
    return BotActionV1(
        target_tick=action.target_tick,
        forward=action.forward,
        side=action.side,
        up=action.up,
        yaw_delta_deg=action.yaw_delta_deg,
        pitch_delta_deg=action.pitch_delta_deg,
        buttons=action.buttons,
        weapon_select=action.weapon_select,
        buy_action=action.buy_action,
        action_valid_mask=0,
    )


def _coerce_action(value: Any) -> tuple[BotActionV1 | None, Any | None]:
    next_state = None
    if isinstance(value, tuple) and len(value) == 2:
        value, next_state = value
    if value is None:
        return None, next_state
    if isinstance(value, BotActionV1):
        return value, next_state
    if isinstance(value, Mapping):
        try:
            return BotActionV1(**dict(value)), next_state
        except (TypeError, ValueError):
            return None, next_state
    return None, next_state


@dataclass(frozen=True)
class OnnxRecurrentStateV1:
    """ONNX Runtime 可序列化的单 Bot 循环状态。"""

    tactical: Any
    action: Any
    tick: Any


class OnnxCpuPolicy:
    """最小 ONNX Runtime CPU 后端，禁止回退到 CUDA provider。"""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
            import numpy as np
        except ImportError as error:
            raise RuntimeError("onnxruntime and numpy are required for CPU runtime") from error
        self._np = np
        self.session = ort.InferenceSession(
            str(Path(model_path)),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.output_names = tuple(item.name for item in self.session.get_outputs())
        self.input_name = self.input_names[0]

    @staticmethod
    def _state_value(state: Any, names: Sequence[str]) -> Any:
        if state is None:
            return None
        for name in names:
            if isinstance(state, Mapping) and name in state:
                return state[name]
            value = getattr(state, name, None)
            if value is not None:
                return value
        return None

    def _state_array(self, value: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        if value is None:
            return self._np.zeros(shape, dtype=dtype)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = self._np.asarray(value, dtype=dtype)
        expected_size = math.prod(shape)
        if array.size != expected_size:
            raise ValueError(f"recurrent state has {array.size} values; expected {expected_size}")
        return array.reshape(shape)

    def infer(self, observations: Sequence[BotObservationV1], states: Mapping[int, Any], server_tick: int) -> Mapping[int, Any]:
        slots = list(states)
        if not slots:
            return {}
        if len(observations) != len(slots):
            raise ValueError("observations must align with policy state slots")
        values = self._np.frombuffer(
            b"".join(observation.to_bytes() for observation in observations),
            dtype=self._np.uint8,
        ).reshape(len(observations), 256).astype(self._np.float32) / 255.0
        feed: dict[str, Any] = {self.input_name: values}
        if "tactical_state" in self.input_names:
            tactical = self._np.stack(
                [self._state_array(self._state_value(states[slot], ("tactical", "tactical_state")), (256,), self._np.float32) for slot in slots]
            )
            action = self._np.stack(
                [self._state_array(self._state_value(states[slot], ("action", "action_state")), (128,), self._np.float32) for slot in slots]
            )
            ticks = self._np.asarray(
                [
                    int(
                        self._state_array(
                            self._state_value(states[slot], ("tick", "tick_state")),
                            (1,),
                            self._np.int64,
                        )[0]
                        if self._state_value(states[slot], ("tick", "tick_state")) is not None
                        else server_tick
                    )
                    for slot in slots
                ],
                dtype=self._np.int64,
            )
            feed.update(
                {
                    "tactical_state": tactical,
                    "action_state": action,
                    "tick": ticks,
                }
            )
        outputs = self.session.run(None, feed)
        if not outputs:
            return {}
        output_map = dict(zip(self.output_names, outputs))
        tensor = output_map.get("action_vector", outputs[0])
        movement_alpha_tensor = output_map.get("movement_alpha")
        movement_beta_tensor = output_map.get("movement_beta")
        mouse_loc_tensor = output_map.get("mouse_loc")
        mouse_scale_tensor = output_map.get("mouse_scale")
        mouse_mix_tensor = output_map.get("mouse_mix_logits")
        button_tensor = output_map.get("button_logits")
        weapon_tensor = output_map.get("weapon_logits")
        buy_tensor = output_map.get("buy_logits")
        if any(value is None for value in (movement_alpha_tensor, movement_beta_tensor, mouse_loc_tensor, mouse_scale_tensor, mouse_mix_tensor, button_tensor, weapon_tensor, buy_tensor)):
            raise ValueError("ONNX actor package must expose all action heads")
        next_tactical = output_map.get("next_tactical_state")
        next_action = output_map.get("next_action_state")
        next_tick = output_map.get("next_tick")
        if next_tactical is None or next_action is None or next_tick is None:
            raise ValueError("ONNX actor package must expose recurrent next-state outputs")
        actions: dict[int, Any] = {}
        for local_index, slot in enumerate(slots):
            row = tensor[local_index]
            action_value = _action_from_onnx_outputs(
                row,
                button_tensor[local_index],
                weapon_tensor[local_index],
                buy_tensor[local_index],
                server_tick=server_tick,
                movement_alpha=movement_alpha_tensor[local_index],
                movement_beta=movement_beta_tensor[local_index],
                mouse_loc=mouse_loc_tensor[local_index],
                mouse_scale=mouse_scale_tensor[local_index],
                mouse_mix_logits=mouse_mix_tensor[local_index],
                random_seed=int(server_tick) * 1009 + int(slot) * 9176,
            )
            next_state = OnnxRecurrentStateV1(
                tactical=self._np.array(next_tactical[local_index], dtype=self._np.float32, copy=True),
                action=self._np.array(next_action[local_index], dtype=self._np.float32, copy=True),
                tick=self._np.array(next_tick[local_index], dtype=self._np.int64, copy=True).reshape(()),
            )
            actions[slot] = (action_value, next_state)
        return actions


class RuntimeService:
    """批量 CPU 推理与逐 Bot 缺帧/故障管理。"""

    def __init__(
        self,
        *,
        transport: SharedMemoryTransportV1,
        policy: Callable[..., Any] | Any | None = None,
        model_path: str | Path | None = None,
        max_bots: int = MAX_BOTS,
        policy_generation: int = 0,
    ) -> None:
        if not 1 <= max_bots <= MAX_BOTS:
            raise ValueError(f"max_bots must be between 1 and {MAX_BOTS}")
        if policy is not None and model_path is not None:
            raise ValueError("provide policy or model_path, not both")
        self.transport = transport
        self.max_bots = max_bots
        self.policy = OnnxCpuPolicy(model_path) if model_path is not None else policy
        if not isinstance(policy_generation, int) or isinstance(policy_generation, bool) or policy_generation < 0:
            raise ValueError("policy_generation must be a non-negative integer")
        self.policy_generation = policy_generation
        self._pending_policy: tuple[int, Any] | None = None
        self._states = [BotRuntimeStateV1() for _ in range(max_bots)]
        self._entity_states: dict[int, BotRuntimeStateV1] = {}
        self._write_sequence = 0
        self._process_failed = False
        self._failure_reason = ""
        self._round_epoch = transport.epoch
        self._last_timing: dict[str, int] = {}

    @property
    def process_failed(self) -> bool:
        return self._process_failed

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    @property
    def last_timing(self) -> dict[str, int]:
        return dict(self._last_timing)

    def bot_state(self, slot: int) -> BotRuntimeStateV1:
        if not 0 <= slot < self.max_bots:
            raise IndexError(f"bot slot must be between 0 and {self.max_bots - 1}")
        return self._states[slot]

    def mark_process_failed(self, reason: str) -> None:
        self._process_failed = True
        self._failure_reason = str(reason)
        for state in self._states:
            state.permanent_fallback = True

    def mark_bot_state_fault(self, slot: int, reason: str = "") -> None:
        state = self.bot_state(slot)
        state.state_fault = True
        state.permanent_fallback = True
        if reason:
            self._failure_reason = f"bot {slot}: {reason}"

    def begin_round(self, *, epoch: int) -> None:
        self.transport.switch_epoch(epoch)
        self._reset_round_state(epoch)

    def stage_policy_generation(self, model_path: str | Path, generation: int) -> None:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("policy generation must be a non-negative integer")
        if generation <= self.policy_generation:
            raise ValueError("staged policy generation must be newer than the active generation")
        policy = OnnxCpuPolicy(model_path)
        self._pending_policy = (generation, policy)

    def activate_policy_generation_at_going_live(self) -> bool:
        if self._pending_policy is None:
            return False
        generation, policy = self._pending_policy
        self.policy = policy
        self.policy_generation = generation
        self._pending_policy = None
        self._reset_round_state(self._round_epoch)
        return True

    def _reset_round_state(self, epoch: int) -> None:
        self._round_epoch = epoch
        self._process_failed = False
        self._failure_reason = ""
        self._write_sequence = 0
        self._states = [BotRuntimeStateV1() for _ in range(self.max_bots)]
        self._entity_states.clear()

    def _sync_round_epoch(self) -> int:
        current_epoch = int(self.transport.refresh_epoch())
        if current_epoch != self._round_epoch:
            self._reset_round_state(current_epoch)
        return current_epoch

    def _resolve_batch_states(self, observations: Sequence[BotObservationV1]) -> None:
        active_entity_ids = {
            entity_id
            for observation in observations
            if (entity_id := _observation_entity_id(observation.to_bytes())) is not None
        }
        if active_entity_ids:
            for entity_id in tuple(self._entity_states):
                if entity_id not in active_entity_ids:
                    del self._entity_states[entity_id]
        else:
            self._entity_states.clear()
        for slot, observation in enumerate(observations):
            entity_id = _observation_entity_id(observation.to_bytes())
            if entity_id is None:
                previous = self._states[slot]
                state = BotRuntimeStateV1(
                    permanent_fallback=previous.permanent_fallback,
                    state_fault=previous.state_fault,
                )
            else:
                state = self._entity_states.get(entity_id)
                if state is None:
                    state = BotRuntimeStateV1(entity_id=entity_id)
                    self._entity_states[entity_id] = state
            self._states[slot] = state

    def _mark_missing(self, slot: int) -> None:
        state = self._states[slot]
        state.consecutive_missing += 1
        state.recent_missing.append(True)
        if state.consecutive_missing >= 32 or state.recent_missing_count >= 32:
            state.permanent_fallback = True

    def _mark_valid(self, slot: int, next_state: Any | None) -> None:
        state = self._states[slot]
        state.consecutive_missing = 0
        state.recent_missing.append(False)
        if next_state is not None:
            state.hidden = next_state

    def _invoke_policy(self, observations: Sequence[BotObservationV1], slots: Sequence[int], server_tick: int) -> Mapping[int, Any]:
        if self.policy is None:
            return {}
        states = {slot: self._states[slot].hidden for slot in slots}
        if hasattr(self.policy, "infer"):
            result = self.policy.infer(observations, states, server_tick)
        else:
            result = self.policy(observations, states, server_tick)
        if result is None:
            return {}
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (Mapping, Sequence)):
            result = result[0]
        if isinstance(result, Mapping):
            return {int(slot): value for slot, value in result.items()}
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            return {slot: value for slot, value in zip(slots, result)}
        raise TypeError("policy must return a slot mapping or an aligned sequence")

    @staticmethod
    def _action_is_current(action: BotActionV1, server_tick: int) -> bool:
        return action.target_tick == server_tick + 1 and all(
            math.isfinite(float(value))
            for value in (action.forward, action.side, action.up, action.yaw_delta_deg, action.pitch_delta_deg)
        )

    def process_observation_batch(self, batch: ObservationBatchV1 | bytes) -> ActionBatchV1:
        if isinstance(batch, (bytes, bytearray, memoryview)):
            batch = ObservationBatchV1.unpack(bytes(batch))
        if not isinstance(batch, ObservationBatchV1):
            raise TypeError("batch must be ObservationBatchV1 or bytes")
        current_epoch = self._sync_round_epoch()
        if batch.epoch != current_epoch:
            raise ValueError(f"epoch mismatch: expected {current_epoch}, got {batch.epoch}")
        if batch.bot_count > self.max_bots:
            raise ValueError(f"runtime configured for at most {self.max_bots} bots")
        observations = tuple(BotObservationV1(value) for value in batch.observations)
        self._resolve_batch_states(observations)
        target_tick = batch.server_tick + 1
        actions: list[BotActionV1] = [neutral_action(target_tick) for _ in range(batch.bot_count)]
        active_slots = [slot for slot in range(batch.bot_count) if not self._states[slot].permanent_fallback]
        inferred: Mapping[int, Any] = {}
        inference_started_ns = time.perf_counter_ns()
        if not self._process_failed and active_slots:
            try:
                inferred = self._invoke_policy(
                    [observations[slot] for slot in active_slots],
                    active_slots,
                    batch.server_tick,
                )
            except Exception as error:
                self.mark_process_failed(str(error))
                inferred = {}
        inference_completed_ns = time.perf_counter_ns()
        for slot in range(batch.bot_count):
            state = self._states[slot]
            if self._process_failed:
                actions[slot] = valve_fallback_action(target_tick)
                continue
            if state.permanent_fallback:
                actions[slot] = valve_fallback_action(target_tick)
                continue
            action, next_state = _coerce_action(inferred.get(slot))
            if action is None or not self._action_is_current(action, batch.server_tick):
                self._mark_missing(slot)
                if self._states[slot].permanent_fallback:
                    actions[slot] = valve_fallback_action(target_tick)
                continue
            if action.action_valid_mask == 0:
                state.permanent_fallback = True
                actions[slot] = valve_fallback_action(target_tick)
                continue
            self._mark_valid(slot, next_state)
            actions[slot] = action
        result = ActionBatchV1(
            epoch=batch.epoch,
            server_tick=batch.server_tick,
            actions=tuple(actions),
            write_sequence=self._write_sequence,
        )
        self._write_sequence += 1
        publish_started_ns = time.perf_counter_ns()
        self.transport.publish_action(result)
        publish_completed_ns = time.perf_counter_ns()
        self._last_timing = {
            "inference_started_ns": inference_started_ns,
            "inference_completed_ns": inference_completed_ns,
            "publish_started_ns": publish_started_ns,
            "publish_completed_ns": publish_completed_ns,
        }
        return result

    def run(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            try:
                self._sync_round_epoch()
                if not self.transport.wait_for_observation(100):
                    continue
                batch = self.transport.try_read_observation()
            except QueueAborted:
                return
            except Exception as error:
                self.mark_process_failed(str(error))
                continue
            if batch is None:
                continue
            self.process_observation_batch(batch)
