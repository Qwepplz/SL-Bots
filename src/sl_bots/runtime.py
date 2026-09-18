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
    INTENT_TASKS,
    IntentV1,
    Phase,
    TACTICAL_MODES,
    TARGET_SLOT_NONE,
    ObservationBatchV1,
)
from .intent import canonicalize_intent, guard_action, guard_friendly_fire


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
    sample_distributions: bool | None = None,
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
    has_movement_distribution = movement_alpha is not None or movement_beta is not None
    if has_movement_distribution:
        if movement_alpha is None or movement_beta is None:
            raise ValueError("ONNX movement distribution outputs must be provided together")
        if len(movement_alpha) != 3 or len(movement_beta) != 3:
            raise ValueError("ONNX movement distribution outputs have invalid dimensions")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (*movement_alpha, *movement_beta)
        ):
            raise ValueError("ONNX movement distribution outputs must be finite and positive")
    distribution_sampling = (
        has_movement_distribution if sample_distributions is None else bool(sample_distributions)
    )
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


@dataclass
class HierarchicalRuntimeStateV1:
    """Numpy runtime state kept independently for each Bot slot."""

    observation_history: Any
    decision_memory: Any
    action_hidden: Any
    cached_intent: Any
    intent: IntentV1 = field(default_factory=IntentV1)
    last_decision_tick: int = -1
    reflection_request_tick: int | None = None


def decision_output_to_intent_embedding_numpy(
    output: Mapping[str, Any],
    index: int,
    *,
    labels: tuple[int, int, int] | None = None,
) -> Any:
    """Encode ONNX decision heads using the canonical IntentV1 contract."""

    import numpy as np

    def row(name: str) -> Any:
        values = np.asarray(output[name], dtype=np.float32)
        if values.ndim == 0 or index < 0 or index >= values.shape[0]:
            raise ValueError(f"decision output {name} does not contain batch index {index}")
        return values[index]

    if labels is None:
        labels = (
            int(row("tactical_mode_logits").argmax()),
            int(row("task_logits").argmax()),
            int(row("target_slot_logits").argmax()),
        )
    if len(labels) != 3:
        raise ValueError("intent labels must contain tactical, task and target classes")
    if not 0 <= int(labels[0]) < len(TACTICAL_MODES):
        raise ValueError("tactical intent label is outside the decision head range")
    if not 0 <= int(labels[1]) < len(INTENT_TASKS):
        raise ValueError("task intent label is outside the decision head range")
    if not 0 <= int(labels[2]) <= TARGET_SLOT_NONE:
        raise ValueError("target intent label is outside the decision head range")
    intent = canonicalize_intent(
        {
            "tactical_mode": TACTICAL_MODES[int(labels[0])],
            "task": INTENT_TASKS[int(labels[1])],
            "goal_position": row("goal_position"),
            "waypoint_position": row("waypoint_position"),
            "facing_yaw_pitch": row("facing_yaw_pitch"),
            "desired_range": row("desired_range"),
            "target_slot": int(labels[2]),
            "aggression": row("aggression"),
            "risk": row("risk"),
            "priority": row("priority"),
            "ttl_ticks": row("ttl_ticks"),
            "confidence": {"all": 1.0},
            "valid_mask": {
                "buy_action": False,
                "utility_action": int(labels[1]) == INTENT_TASKS.index("use_utility"),
                "weapon_select": True,
            },
        }
    )
    return np.asarray(intent.to_embedding(128), dtype=np.float32)


class HierarchicalRuntimeService:
    """CPU double-session runtime for the 16 Hz decision/128 Hz action split."""

    def __init__(
        self,
        *,
        decision_session: Any | None = None,
        action_session: Any | None = None,
        decision_model_path: str | Path | None = None,
        action_model_path: str | Path | None = None,
        max_bots: int = MAX_BOTS,
        tick_deadline_ms: float = 1000.0 / 128.0,
        stochastic_actions: bool = False,
        stochastic_decisions: bool = False,
    ) -> None:
        if not 1 <= max_bots <= MAX_BOTS:
            raise ValueError(f"max_bots must be between 1 and {MAX_BOTS}")
        if decision_session is not None and decision_model_path is not None:
            raise ValueError("provide decision_session or decision_model_path, not both")
        if action_session is not None and action_model_path is not None:
            raise ValueError("provide action_session or action_model_path, not both")
        if decision_session is None:
            decision_session = self._load_cpu_session(decision_model_path, "decision_model_path")
        if action_session is None:
            action_session = self._load_cpu_session(action_model_path, "action_model_path")
        if decision_session is action_session:
            raise ValueError("decision and action must use independent ONNX Runtime sessions")
        if tick_deadline_ms <= 0.0:
            raise ValueError("tick_deadline_ms must be positive")
        if not isinstance(stochastic_actions, bool):
            raise TypeError("stochastic_actions must be a boolean")
        if not isinstance(stochastic_decisions, bool):
            raise TypeError("stochastic_decisions must be a boolean")
        import numpy as np

        self._np = np
        self.decision_session = decision_session
        self.action_session = action_session
        self.max_bots = max_bots
        self.tick_deadline_ms = float(tick_deadline_ms)
        self.stochastic_actions = stochastic_actions
        self.stochastic_decisions = stochastic_decisions
        self._decision_input_names = tuple(item.name for item in decision_session.get_inputs())
        self._decision_output_names = tuple(item.name for item in decision_session.get_outputs())
        self._action_input_names = tuple(item.name for item in action_session.get_inputs())
        self._action_output_names = tuple(item.name for item in action_session.get_outputs())
        self._states = [self._new_state() for _ in range(max_bots)]
        self._last_tick = -1
        self._decision_latency_ms: list[float] = []
        self._action_latency_ms: list[float] = []
        self._tick_latency_ms: list[float] = []
        self._last_action_elapsed_ms = 0.0
        self._deadline_miss_count = 0
        self._permission_violations = 0
        self._last_decision_actions: tuple[tuple[int, int, int] | None, ...] = tuple(
            None for _ in range(max_bots)
        )
        self._last_decision_log_probs: tuple[float | None, ...] = tuple(
            None for _ in range(max_bots)
        )

    @staticmethod
    def _load_cpu_session(model_path: str | Path | None, name: str) -> Any:
        if model_path is None:
            raise ValueError(f"{name} is required when a session is not supplied")
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency boundary
            raise RuntimeError("onnxruntime is required for hierarchical runtime") from error
        session_options = ort.SessionOptions()
        # The action branch runs at 128 Hz and is intentionally a small batch.
        # A large ORT thread pool adds scheduling jitter across the game server
        # and multiple farm instances, so use deterministic sequential CPU
        # execution for both independent hierarchical sessions.
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(Path(model_path)),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        providers = tuple(session.get_providers()) if hasattr(session, "get_providers") else ()
        if providers and providers != ("CPUExecutionProvider",):
            raise RuntimeError("hierarchical runtime must use independent CPU ONNX sessions")
        return session

    def _new_state(self) -> HierarchicalRuntimeStateV1:
        return HierarchicalRuntimeStateV1(
            observation_history=self._np.zeros((32, 256), dtype=self._np.float32),
            decision_memory=self._np.zeros((32, 512), dtype=self._np.float32),
            action_hidden=self._np.zeros((384,), dtype=self._np.float32),
            cached_intent=self._np.zeros((128,), dtype=self._np.float32),
        )

    def bot_state(self, slot: int) -> HierarchicalRuntimeStateV1:
        if not 0 <= slot < self.max_bots:
            raise IndexError(f"bot slot must be between 0 and {self.max_bots - 1}")
        return self._states[slot]

    def _reset_states(self) -> None:
        self._states = [self._new_state() for _ in range(self.max_bots)]
        self._last_tick = -1
        self._last_decision_actions = tuple(None for _ in range(self.max_bots))
        self._last_decision_log_probs = tuple(None for _ in range(self.max_bots))

    def reset_round(self) -> None:
        self._reset_states()

    def begin_round(self) -> None:
        self.reset_round()

    def change_side(self) -> None:
        self._reset_states()

    def end_match(self) -> None:
        self._reset_states()

    def backup_state(self) -> list[HierarchicalRuntimeStateV1]:
        import copy

        return copy.deepcopy(self._states)

    def restore_backup(self, backup: Sequence[HierarchicalRuntimeStateV1]) -> None:
        import copy

        if len(backup) != self.max_bots:
            raise ValueError("hierarchical runtime backup does not match max_bots")
        self._states = copy.deepcopy(list(backup))
        self._last_tick = max((state.last_decision_tick for state in self._states), default=-1)

    def request_reflection(self, slot: int, *, requested_at_tick: int | None = None) -> None:
        state = self.bot_state(slot)
        tick = self._last_tick if requested_at_tick is None else int(requested_at_tick)
        if tick < 0:
            raise ValueError("reflection request needs a non-negative tick")
        state.reflection_request_tick = tick

    @staticmethod
    def _append_observation(state: HierarchicalRuntimeStateV1, observation: BotObservationV1, np: Any) -> Any:
        values = np.frombuffer(observation.to_bytes(), dtype=np.uint8).astype(np.float32) / 255.0
        state.observation_history[:-1] = state.observation_history[1:]
        state.observation_history[-1] = values
        return values

    @staticmethod
    def _decision_to_intent(
        output: Mapping[str, Any],
        index: int,
        labels: tuple[int, int, int] | None = None,
        *,
        buy_allowed: bool = True,
    ) -> IntentV1:
        if labels is None:
            labels = (
                int(output["tactical_mode_logits"][index].argmax()),
                int(output["task_logits"][index].argmax()),
                int(output["target_slot_logits"][index].argmax()),
            )
        mode = TACTICAL_MODES[int(labels[0])]
        task = INTENT_TASKS[int(labels[1])]
        target_slot = int(labels[2])
        if target_slot > TARGET_SLOT_NONE:
            target_slot = TARGET_SLOT_NONE
        valid_mask = {
            # The bridge enforces the actual buy phase.  The intent layer must
            # not permanently disable the buy head before that deployment
            # boundary gets a chance to apply its phase rules.
            "buy_action": bool(buy_allowed),
            "utility_action": task == "use_utility",
            "weapon_select": True,
        }
        return canonicalize_intent(
            {
                "tactical_mode": mode,
                "task": task,
                "goal_position": output["goal_position"][index],
                "waypoint_position": output["waypoint_position"][index],
                "facing_yaw_pitch": output["facing_yaw_pitch"][index],
                "desired_range": output["desired_range"][index],
                "target_slot": target_slot,
                "aggression": output["aggression"][index],
                "risk": output["risk"][index],
                "priority": output["priority"][index],
                "ttl_ticks": output["ttl_ticks"][index],
                "confidence": {"all": 1.0},
                "valid_mask": valid_mask,
            }
        )

    @staticmethod
    def _buy_allowed(observation: BotObservationV1) -> bool:
        """Allow the buy head on a valid Get5 phase; the bridge gates the buy window."""

        try:
            from .observation_projection import decode_observation

            phase = decode_observation(observation).phase
        except (TypeError, ValueError, struct.error):
            return False
        # CS:GO exposes round freezetime through the live phase in this wire
        # contract.  The SourcePawn bridge suppresses/duplicates buy commands
        # according to the actual engine window, so disabling the head for all
        # LIVE observations would make normal round purchases impossible.
        return phase in {Phase.WARMUP, Phase.KNIFE, Phase.LIVE}

    def _run_decision(
        self,
        slots: Sequence[int],
        server_tick: int,
        observations: Sequence[BotObservationV1] | None = None,
    ) -> None:
        if not slots:
            return
        np = self._np
        observation_history = np.stack([self._states[slot].observation_history for slot in slots])
        previous_intent = np.stack([self._states[slot].cached_intent for slot in slots])
        memory = np.stack([self._states[slot].decision_memory for slot in slots])
        feed = {
            self._decision_input_names[0]: observation_history,
            self._decision_input_names[1]: previous_intent,
            self._decision_input_names[2]: memory,
        }
        started = time.perf_counter_ns()
        values = self.decision_session.run(None, feed)
        completed = time.perf_counter_ns()
        self._decision_latency_ms.append((completed - started) / 1_000_000.0)
        output = dict(zip(self._decision_output_names, values))
        required = {
            "tactical_mode_logits",
            "task_logits",
            "goal_position",
            "waypoint_position",
            "facing_yaw_pitch",
            "desired_range",
            "target_slot_logits",
            "aggression",
            "risk",
            "priority",
            "ttl_ticks",
        }
        if not required.issubset(output):
            raise ValueError("decision session does not expose the structured intent outputs")
        decision_actions = list(self._last_decision_actions)
        decision_log_probs = list(self._last_decision_log_probs)
        for local_index, slot in enumerate(slots):
            state = self._states[slot]
            tactical_logits = np.asarray(output["tactical_mode_logits"][local_index], dtype=np.float64)
            task_logits = np.asarray(output["task_logits"][local_index], dtype=np.float64)
            target_logits = np.asarray(output["target_slot_logits"][local_index], dtype=np.float64)
            logits_by_head = (tactical_logits, task_logits, target_logits)

            if self.stochastic_decisions:
                # Rollouts used by MAPPO must record the probability under the
                # policy that actually selected the action.  Keep sampling
                # reproducible per Bot/Tick so replay and diagnostics remain
                # deterministic while still representing a categorical policy.
                rng = np.random.default_rng(
                    int(server_tick) * 1009 + int(slot) * 9176 + 0x5EED
                )
                sampled: list[int] = []
                log_prob = 0.0
                for logits in logits_by_head:
                    shifted = logits - np.max(logits)
                    probabilities = np.exp(shifted)
                    probabilities /= np.sum(probabilities)
                    label = int(rng.choice(len(probabilities), p=probabilities))
                    sampled.append(label)
                    log_prob += float(np.log(max(float(probabilities[label]), 1e-12)))
                labels = (sampled[0], sampled[1], sampled[2])
            else:
                # The deterministic runtime behavior is an argmax point mass.
                # Its behavior probability is therefore exactly one; recording
                # the softmax probability here would make PPO's old_log_prob
                # describe a policy that did not select the action.
                labels = tuple(int(logits.argmax()) for logits in logits_by_head)
                log_prob = 0.0

            buy_allowed = True
            if observations is not None:
                buy_allowed = self._buy_allowed(observations[slot])
            state.intent = self._decision_to_intent(
                output,
                local_index,
                labels,
                buy_allowed=buy_allowed,
            )
            state.cached_intent = decision_output_to_intent_embedding_numpy(
                output,
                local_index,
                labels=labels,
            )
            next_memory = output.get("next_decision_memory")
            if next_memory is not None:
                state.decision_memory = np.asarray(next_memory[local_index], dtype=np.float32).copy()
            state.last_decision_tick = int(server_tick)
            state.reflection_request_tick = None
            decision_actions[slot] = labels
            decision_log_probs[slot] = log_prob
        self._last_decision_actions = tuple(decision_actions)
        self._last_decision_log_probs = tuple(decision_log_probs)

    def last_decision_behavior(
        self,
        count: int | None = None,
    ) -> tuple[tuple[tuple[int, int, int] | None, ...], tuple[float | None, ...]]:
        """Return the behavior decision recorded for the most recent tick."""

        if count is None:
            count = self.max_bots
        if not 0 <= count <= self.max_bots:
            raise ValueError("decision behavior count is outside runtime capacity")
        return self._last_decision_actions[:count], self._last_decision_log_probs[:count]

    def _run_action(self, observations: Sequence[BotObservationV1], server_tick: int) -> tuple[BotActionV1, ...]:
        np = self._np
        if not observations:
            self._last_action_elapsed_ms = 0.0
            return ()
        local_observation = np.stack(
            [np.frombuffer(observation.to_bytes(), dtype=np.uint8).astype(np.float32) / 255.0 for observation in observations]
        )
        cached_intent = np.stack([self._states[index].cached_intent for index in range(len(observations))])
        action_hidden = np.stack([self._states[index].action_hidden for index in range(len(observations))])
        feed = {
            self._action_input_names[0]: local_observation,
            self._action_input_names[1]: cached_intent,
            self._action_input_names[2]: action_hidden,
        }
        started = time.perf_counter_ns()
        values = self.action_session.run(None, feed)
        completed = time.perf_counter_ns()
        self._last_action_elapsed_ms = (completed - started) / 1_000_000.0
        self._action_latency_ms.append(self._last_action_elapsed_ms)
        output = dict(zip(self._action_output_names, values))
        required = {
            "action_vector",
            "movement_alpha",
            "movement_beta",
            "mouse_loc",
            "mouse_scale",
            "mouse_mix_logits",
            "button_logits",
            "weapon_logits",
            "buy_logits",
            "next_action_hidden",
        }
        if not required.issubset(output):
            raise ValueError("action session does not expose all action heads and next state")
        actions: list[BotActionV1] = []
        for index in range(len(observations)):
            action = _action_from_onnx_outputs(
                output["action_vector"][index],
                output["button_logits"][index],
                output["weapon_logits"][index],
                output["buy_logits"][index],
                server_tick=server_tick,
                movement_alpha=output["movement_alpha"][index],
                movement_beta=output["movement_beta"][index],
                mouse_loc=output["mouse_loc"][index],
                mouse_scale=output["mouse_scale"][index],
                mouse_mix_logits=output["mouse_mix_logits"][index],
                random_seed=server_tick * 1009 + index * 9176,
                sample_distributions=self.stochastic_actions,
            )
            guarded = guard_action(
                self._states[index].intent,
                action,
                observation=observations[index],
            )
            guarded = guard_friendly_fire(observations[index], guarded)
            if guarded != action:
                self._permission_violations += 1
            self._states[index].action_hidden = np.asarray(
                output["next_action_hidden"][index], dtype=np.float32
            ).copy()
            actions.append(guarded)
        return tuple(actions)

    @staticmethod
    def _percentiles(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        import numpy as np

        p50, p95, p99 = np.percentile(np.asarray(values, dtype=np.float64), [50, 95, 99])
        return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}

    def metrics(self) -> dict[str, Any]:
        return {
            "decision_ms": self._percentiles(self._decision_latency_ms),
            "action_ms": self._percentiles(self._action_latency_ms),
            "tick_ms": self._percentiles(self._tick_latency_ms),
            "total_tick_deadline_miss": self._deadline_miss_count,
            "permission_violations": self._permission_violations,
        }

    def process_tick(
        self,
        observations: Sequence[BotObservationV1],
        *,
        server_tick: int,
        important_events: Sequence[Any] = (),
    ) -> tuple[BotActionV1, ...]:
        if server_tick < 0:
            raise ValueError("server_tick must be non-negative")
        if len(observations) > self.max_bots:
            raise ValueError(f"runtime configured for at most {self.max_bots} bots")
        observations = tuple(BotObservationV1(value.to_bytes()) for value in observations)
        started = time.perf_counter_ns()
        self._last_decision_actions = tuple(None for _ in range(self.max_bots))
        self._last_decision_log_probs = tuple(None for _ in range(self.max_bots))
        for slot, observation in enumerate(observations):
            self._append_observation(self._states[slot], observation, self._np)
        from .hierarchical_model import decision_refresh_due

        due = [
            slot
            for slot in range(len(observations))
            if decision_refresh_due(
                server_tick,
                self._states[slot].last_decision_tick,
                self._states[slot].intent.ttl_ticks,
                important_event=bool(important_events),
                reflection_request_tick=self._states[slot].reflection_request_tick,
            )
        ]
        self._run_decision(due, server_tick, observations)
        actions = self._run_action(observations, server_tick)
        completed = time.perf_counter_ns()
        elapsed_ms = (completed - started) / 1_000_000.0
        self._tick_latency_ms.append(elapsed_ms)
        # Decision and action are synchronous on a tick.  A decision refresh
        # therefore delays the action publication and must be included in the
        # hard 128 Hz deadline measurement.
        if elapsed_ms > self.tick_deadline_ms:
            self._deadline_miss_count += 1
        self._last_tick = server_tick
        return actions


class HierarchicalSharedMemoryRuntimeV1:
    """接入 SourceMod 共享内存的双模型运行时适配层。

    ``HierarchicalRuntimeService`` 只负责分层 ONNX 推理和每 Bot 状态；本类
    负责 Protocol V1 批次、Get5 边界、动作发布以及进程级故障回退。这样旧的
    单模型 ``RuntimeService`` 兼容路径可以继续服务 production 命令，而
    test-only 分层模型不会被错误地塞进旧 actor 输入契约。
    """

    def __init__(
        self,
        *,
        transport: Any,
        runtime: HierarchicalRuntimeService | None = None,
        decision_session: Any | None = None,
        action_session: Any | None = None,
        decision_model_path: str | Path | None = None,
        action_model_path: str | Path | None = None,
        max_bots: int = MAX_BOTS,
        tick_deadline_ms: float = 1000.0 / 128.0,
        stochastic_decisions: bool = False,
    ) -> None:
        if runtime is not None and any(
            value is not None
            for value in (
                decision_session,
                action_session,
                decision_model_path,
                action_model_path,
            )
        ):
            raise ValueError("provide runtime or hierarchical model/session inputs, not both")
        if not 1 <= max_bots <= MAX_BOTS:
            raise ValueError(f"max_bots must be between 1 and {MAX_BOTS}")
        if runtime is None:
            runtime = HierarchicalRuntimeService(
                decision_session=decision_session,
                action_session=action_session,
                decision_model_path=decision_model_path,
                action_model_path=action_model_path,
                max_bots=max_bots,
                tick_deadline_ms=tick_deadline_ms,
                stochastic_decisions=stochastic_decisions,
            )
        if not isinstance(runtime, HierarchicalRuntimeService):
            raise TypeError("runtime must be a HierarchicalRuntimeService")
        if runtime.max_bots < max_bots:
            raise ValueError("shared-memory max_bots cannot exceed hierarchical runtime capacity")
        self.transport = transport
        self.runtime = runtime
        self.max_bots = int(max_bots)
        self._round_epoch = int(transport.epoch)
        self._write_sequence = 0
        self._process_failed = False
        self._failure_reason = ""
        self._last_timing: dict[str, int] = {}
        self._pending_events: list[Any] = []

    @property
    def process_failed(self) -> bool:
        return self._process_failed

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    @property
    def last_timing(self) -> dict[str, int]:
        return dict(self._last_timing)

    def metrics(self) -> dict[str, Any]:
        metrics = dict(self.runtime.metrics())
        metrics.update(
            {
                "process_failed": self.process_failed,
                "failure_reason": self.failure_reason,
            }
        )
        return metrics

    def bot_state(self, slot: int) -> HierarchicalRuntimeStateV1:
        return self.runtime.bot_state(slot)

    def mark_process_failed(self, reason: str) -> None:
        self._process_failed = True
        self._failure_reason = str(reason)

    def apply_boundaries(self, boundaries: Sequence[Any]) -> None:
        """Apply Get5 control boundaries before the next observation tick."""

        for boundary in boundaries:
            self._pending_events.append(boundary)
            kind = str(getattr(boundary, "kind", ""))
            reset_hidden = bool(getattr(boundary, "reset_hidden", False))
            terminal = bool(getattr(boundary, "terminal", False))
            if kind in {"map_end", "series_end"} or terminal and kind in {
                "map_end",
                "series_end",
            }:
                self.runtime.end_match()
            elif reset_hidden or kind in {
                "round_end",
                "halftime",
                "overtime_start",
                "backup_restore",
            }:
                self.runtime.reset_round()

    def _sync_round_epoch(self) -> int:
        current_epoch = int(self.transport.refresh_epoch())
        if current_epoch != self._round_epoch:
            self.runtime.reset_round()
            self._round_epoch = current_epoch
            self._write_sequence = 0
            self._process_failed = False
            self._failure_reason = ""
        return current_epoch

    def process_observation_batch(self, batch: ObservationBatchV1 | bytes) -> ActionBatchV1 | None:
        if isinstance(batch, (bytes, bytearray, memoryview)):
            batch = ObservationBatchV1.unpack(bytes(batch))
        if not isinstance(batch, ObservationBatchV1):
            raise TypeError("batch must be ObservationBatchV1 or bytes")
        current_epoch = self._sync_round_epoch()
        if batch.epoch != current_epoch:
            # The bridge can leave one observation from the previous round in
            # the ring while the native epoch has already advanced.  It is
            # stale work, not an inference/runtime failure: the epoch sync
            # above has already reset hidden state and action sequencing.
            return None
        if batch.bot_count > self.max_bots:
            raise ValueError(f"runtime configured for at most {self.max_bots} bots")

        observations = tuple(BotObservationV1(value) for value in batch.observations)
        target_tick = int(batch.server_tick) + 1
        inference_started_ns = time.perf_counter_ns()
        actions: tuple[BotActionV1, ...]
        pending_events = tuple(self._pending_events)
        self._pending_events.clear()
        if self._process_failed:
            actions = tuple(valve_fallback_action(target_tick) for _ in observations)
        else:
            try:
                actions = tuple(
                    self.runtime.process_tick(
                        observations,
                        server_tick=int(batch.server_tick),
                        important_events=pending_events,
                    )
                )
                if len(actions) != batch.bot_count or any(
                    not isinstance(action, BotActionV1) for action in actions
                ):
                    raise ValueError("hierarchical runtime returned an invalid action batch")
            except Exception as error:
                self.mark_process_failed(str(error))
                actions = tuple(valve_fallback_action(target_tick) for _ in observations)
        inference_completed_ns = time.perf_counter_ns()
        result = ActionBatchV1(
            epoch=batch.epoch,
            server_tick=batch.server_tick,
            actions=actions,
            write_sequence=self._write_sequence,
            decision_actions=(
                tuple(None for _ in actions)
                if self._process_failed
                else self.runtime.last_decision_behavior(batch.bot_count)[0]
            ),
            decision_log_probs=(
                tuple(None for _ in actions)
                if self._process_failed
                else self.runtime.last_decision_behavior(batch.bot_count)[1]
            ),
        )
        self._write_sequence += 1
        publish_started_ns = time.perf_counter_ns()
        try:
            self.transport.publish_action(result)
        except ValueError as error:
            # Get5 may advance the native IPC epoch between observation read and
            # action publish (map/round/live boundary).  The computed action is
            # stale for that epoch; discard it and reset state for the next
            # observation instead of turning a normal boundary into a permanent
            # inference fallback.
            if "epoch" not in str(error).lower():
                raise
            self._sync_round_epoch()
            return None
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
                while batch is not None:
                    candidate = self.transport.try_read_observation()
                    if candidate is None:
                        break
                    batch = candidate
            except QueueAborted:
                return
            except Exception as error:
                self.mark_process_failed(str(error))
                continue
            if batch is not None:
                try:
                    self.process_observation_batch(batch)
                except QueueAborted:
                    # Shutdown may abort the native ring between the observation
                    # read and action publish.  Treat that as an orderly worker
                    # stop instead of leaking an exception from the runtime
                    # thread into the launcher.
                    return
                except QueueFull as error:
                    # Under an overloaded farm the bridge may not consume an
                    # action slot before the next observation arrives.  Record
                    # the overload for the capacity gate and keep the worker
                    # alive so the next slot can recover instead of leaking a
                    # thread exception.
                    self.mark_process_failed(str(error))


HierarchicalRuntimeV1 = HierarchicalRuntimeService


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
                while batch is not None:
                    candidate = self.transport.try_read_observation()
                    if candidate is None:
                        break
                    batch = candidate
            except QueueAborted:
                return
            except Exception as error:
                self.mark_process_failed(str(error))
                continue
            if batch is None:
                continue
            self.process_observation_batch(batch)
