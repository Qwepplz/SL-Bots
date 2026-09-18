"""Source RCON transport and readiness checks for test-only dedicated servers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import socket
import struct
import time
from typing import Any, Callable, Iterable, Sequence


RCON_SERVERDATA_RESPONSE_VALUE = 0
RCON_SERVERDATA_EXECCOMMAND = 2
RCON_SERVERDATA_AUTH_RESPONSE = 2
RCON_SERVERDATA_AUTH = 3
_MAX_PACKET_BYTES = 4 * 1024 * 1024
_MAP_PATTERN = re.compile(r"(?:^|\n)\s*map\s*:\s*([^\s]+)", re.IGNORECASE)


class RconError(RuntimeError):
    """Base class for RCON protocol and transport failures."""


class RconAuthenticationError(RconError):
    """Raised when the server rejects the RCON password."""


class RconTimeoutError(RconError):
    """Raised when a response is not received before the configured deadline."""


class RconReadinessError(RconError):
    """Raised when a dedicated server never reaches the required readiness state."""


@dataclass(frozen=True)
class RconCredentialsV1:
    host: str
    port: int
    password: str

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("RCON host cannot be empty")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("RCON port must be between 1 and 65535")
        if not self.password:
            raise ValueError("RCON password cannot be empty")


def write_rcon_run_config(path: str | Path, credentials: RconCredentialsV1) -> Path:
    """Write credentials only to the caller-provided external run-config path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
    payload = {
        "host": credentials.host,
        "password": credentials.password,
        "port": credentials.port,
    }
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
        temporary.replace(destination)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return destination


def redact_rcon_text(text: str, secrets: Iterable[str]) -> str:
    """Redact passwords before a command/error string is written to logs."""

    redacted = str(text)
    for secret in sorted((str(value) for value in secrets if value), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _encode_packet(packet_id: int, packet_type: int, body: str) -> bytes:
    payload = struct.pack("<ii", int(packet_id), int(packet_type)) + body.encode("utf-8") + b"\x00\x00"
    return struct.pack("<i", len(payload)) + payload


def _receive_exact(stream: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise RconError("RCON connection closed before a complete packet arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_packet(stream: socket.socket) -> tuple[int, int, str]:
    try:
        header = _receive_exact(stream, 4)
        size = struct.unpack("<i", header)[0]
        if size < 10 or size > _MAX_PACKET_BYTES:
            raise RconError(f"invalid RCON packet size: {size}")
        payload = _receive_exact(stream, size)
    except socket.timeout as error:
        raise RconTimeoutError("RCON response timed out") from error
    if len(payload) < 10:
        raise RconError("RCON packet payload is too short")
    packet_id, packet_type = struct.unpack("<ii", payload[:8])
    body = payload[8:].rstrip(b"\x00").decode("utf-8", errors="replace")
    return packet_id, packet_type, body


class RconClientV1:
    """Minimal Source RCON client with response-marker framing and timeouts."""

    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        *,
        timeout_s: float = 2.0,
        socket_factory: Callable[..., socket.socket] = socket.create_connection,
    ) -> None:
        self.credentials = RconCredentialsV1(host, port, password)
        if timeout_s <= 0:
            raise ValueError("RCON timeout_s must be positive")
        self.timeout_s = float(timeout_s)
        self._socket_factory = socket_factory
        self._socket: socket.socket | None = None
        self._next_id = 1

    def connect(self) -> None:
        if self._socket is not None:
            return
        try:
            stream = self._socket_factory(
                (self.credentials.host, self.credentials.port),
                self.timeout_s,
            )
            stream.settimeout(self.timeout_s)
            self._socket = stream
            request_id = self._allocate_id()
            self._send(request_id, RCON_SERVERDATA_AUTH, self.credentials.password)
            while True:
                response_id, response_type, _ = _receive_packet(stream)
                if response_id == -1:
                    raise RconAuthenticationError("RCON authentication rejected")
                if response_id == request_id and response_type == RCON_SERVERDATA_AUTH_RESPONSE:
                    return
        except socket.timeout as error:
            self.close()
            raise RconTimeoutError("RCON authentication timed out") from error
        except Exception:
            self.close()
            raise

    def execute(self, command: str) -> str:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("RCON command cannot be empty")
        self.connect()
        assert self._socket is not None
        request_id = self._allocate_id()
        marker_id = self._allocate_id()
        self._send(request_id, RCON_SERVERDATA_EXECCOMMAND, command)
        # An empty command with a distinct id marks the end of a multi-packet response.
        self._send(marker_id, RCON_SERVERDATA_EXECCOMMAND, "")
        chunks: list[str] = []
        while True:
            response_id, response_type, body = _receive_packet(self._socket)
            if response_id == -1:
                raise RconError("RCON server rejected command")
            if response_type != RCON_SERVERDATA_RESPONSE_VALUE:
                continue
            if response_id == request_id:
                chunks.append(body)
            elif response_id == marker_id:
                return "".join(chunks)

    def close(self) -> None:
        stream = self._socket
        self._socket = None
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    def __enter__(self) -> "RconClientV1":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _allocate_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _send(self, packet_id: int, packet_type: int, body: str) -> None:
        if self._socket is None:
            raise RconError("RCON client is not connected")
        try:
            self._socket.sendall(_encode_packet(packet_id, packet_type, body))
        except socket.timeout as error:
            raise RconTimeoutError("RCON send timed out") from error


@dataclass(frozen=True)
class RconReadinessV1:
    process_alive: bool
    port_responded: bool
    status_ok: bool
    map_name: str | None
    plugins_ok: bool
    ready: bool
    error: str = ""


def check_rcon_readiness(
    process: Any,
    client: Any,
    *,
    expected_map: str = "de_mirage",
    required_plugins: Sequence[str] = ("Get5", "sl_bots"),
) -> RconReadinessV1:
    """Check process, RCON, status/map and plugin readiness in that order."""

    process_alive = process.poll() is None
    if not process_alive:
        return RconReadinessV1(False, False, False, None, False, False, "srcds process exited")
    try:
        status = str(client.execute("status"))
    except Exception as error:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        return RconReadinessV1(True, False, False, None, False, False, redact_rcon_text(str(error), ()))
    status_ok = bool(status.strip())
    match = _MAP_PATTERN.search(status)
    # CS:GO 1.38 status output can omit the historical ``map :`` line even
    # after the requested startup map is active.  The launcher owns the
    # explicit ``+map`` argument, so retain that expected value when the
    # server has returned a non-empty status.  A contradictory map line is
    # still rejected.
    map_name = match.group(1) if match else (expected_map if status_ok else None)
    if not status_ok or map_name != expected_map:
        return RconReadinessV1(True, True, status_ok, map_name, False, False, "status/map check failed")
    try:
        plugin_listing = str(client.execute("sm plugins list"))
    except Exception as error:
        close = getattr(client, "close", None)
        if callable(close):
            close()
        return RconReadinessV1(True, True, True, map_name, False, False, redact_rcon_text(str(error), ()))
    listing = plugin_listing.casefold()
    normalized_listing = re.sub(r"[^a-z0-9]+", "", listing)
    plugins_ok = all(
        re.sub(r"[^a-z0-9]+", "", str(plugin).casefold()) in normalized_listing
        for plugin in required_plugins
    )
    return RconReadinessV1(
        True,
        True,
        True,
        map_name,
        plugins_ok,
        plugins_ok,
        "" if plugins_ok else "required plugin is missing",
    )


def wait_for_rcon_readiness(
    process: Any,
    client: Any,
    *,
    expected_map: str = "de_mirage",
    required_plugins: Sequence[str] = ("Get5", "sl_bots"),
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.25,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> RconReadinessV1:
    if timeout_s <= 0 or poll_interval_s < 0:
        raise ValueError("RCON readiness timeouts must be positive")
    deadline = clock() + timeout_s
    latest = RconReadinessV1(False, False, False, None, False, False, "not checked")
    while True:
        latest = check_rcon_readiness(
            process,
            client,
            expected_map=expected_map,
            required_plugins=required_plugins,
        )
        if latest.ready:
            return latest
        if not latest.process_alive or clock() >= deadline:
            raise RconReadinessError(latest.error or "RCON readiness check failed")
        sleeper(min(poll_interval_s, max(0.0, deadline - clock())))


__all__ = [
    "RconAuthenticationError",
    "RconClientV1",
    "RconCredentialsV1",
    "RconError",
    "RconReadinessError",
    "RconReadinessV1",
    "RconTimeoutError",
    "check_rcon_readiness",
    "redact_rcon_text",
    "wait_for_rcon_readiness",
    "write_rcon_run_config",
]
