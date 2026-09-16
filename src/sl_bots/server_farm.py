from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence


_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_INSTANCE_PATTERN = re.compile(r"[0-9]{2}\Z")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GET5_TEMPLATE = _REPOSITORY_ROOT / "config" / "get5" / "selfplay_mr12.template.json"


@dataclass(frozen=True)
class ServerInstanceSpecV1:
    instance_id: str
    game_port: int
    client_port: int
    tv_port: int
    steam_port: int
    ipc_name: str
    get5_config_path: Path
    log_dir: Path

    @property
    def ports(self) -> tuple[int, int, int, int]:
        return self.game_port, self.client_port, self.tv_port, self.steam_port


def _validate_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _COMPONENT_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a single safe path component")
    return value


def server_get5_path(server_root: Path, run_id: str, instance_id: str) -> Path:
    root = Path(server_root).resolve(strict=False)
    safe_run_id = _validate_component(run_id, "run_id")
    if not isinstance(instance_id, str) or not _INSTANCE_PATTERN.fullmatch(instance_id):
        raise ValueError("instance_id must be a two-digit value")
    path = root / "csgo" / "cfg" / "get5" / "slbots" / safe_run_id / f"instance-{instance_id}.json"
    resolved = path.resolve(strict=False)
    if root not in resolved.parents:
        raise ValueError("Get5 config path escaped server root")
    return resolved


def build_server_specs(
    count: int,
    run_id: str,
    server_root: Path,
    output_root: Path,
) -> tuple[ServerInstanceSpecV1, ...]:
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 4:
        raise ValueError("server count must be between 1 and 4")
    safe_run_id = _validate_component(run_id, "run_id")
    server_root = Path(server_root)
    output_root = Path(output_root)
    return tuple(
        ServerInstanceSpecV1(
            instance_id=f"{index:02d}",
            game_port=27100 + index,
            client_port=27200 + index,
            tv_port=27300 + index,
            steam_port=27400 + index,
            ipc_name=f"SLBots_de_mirage_{index:02d}",
            get5_config_path=server_get5_path(server_root, safe_run_id, f"{index:02d}"),
            log_dir=(output_root / "logs" / f"instance-{index:02d}").resolve(strict=False),
        )
        for index in range(1, count + 1)
    )


def _replace_match_id(value: Any, match_id: str) -> Any:
    if isinstance(value, str):
        return value.replace("{MATCH_ID}", match_id).replace("{MATCHID}", match_id)
    if isinstance(value, list):
        return [_replace_match_id(item, match_id) for item in value]
    if isinstance(value, dict):
        return {key: _replace_match_id(item, match_id) for key, item in value.items()}
    return value


def render_get5_config(
    spec: ServerInstanceSpecV1,
    match_id: str,
    *,
    generation: int,
    template_path: Path | None = None,
) -> dict[str, Any]:
    if not match_id or not isinstance(match_id, str):
        raise ValueError("match_id is required")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    source = _GET5_TEMPLATE if template_path is None else Path(template_path)
    data = _replace_match_id(json.loads(source.read_text(encoding="utf-8")), match_id)
    if not isinstance(data, dict):
        raise ValueError("Get5 template root must be an object")
    data["matchid"] = match_id
    cvars = data.setdefault("cvars", {})
    if not isinstance(cvars, dict):
        raise ValueError("Get5 template cvars must be an object")
    cvars["sl_bots_instance_id"] = spec.instance_id
    cvars["sl_bots_policy_generation"] = str(generation)
    cvars["get5_server_id"] = f"slbots-{spec.instance_id}"
    return data


def _hidden_popen_options() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


class DedicatedServerFarm:
    def __init__(
        self,
        server_root: Path,
        output_root: Path,
        run_id: str,
        *,
        max_servers: int = 4,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(max_servers, int) or isinstance(max_servers, bool) or not 1 <= max_servers <= 4:
            raise ValueError("max_servers must be between 1 and 4")
        self.server_root = Path(server_root).resolve(strict=False)
        self.output_root = Path(output_root).resolve(strict=False)
        self.run_id = _validate_component(run_id, "run_id")
        self.max_servers = max_servers
        self._process_factory = subprocess.Popen if process_factory is None else process_factory
        self._processes: dict[str, Any] = {}
        self._log_handles: dict[str, tuple[Any, Any]] = {}
        self._specs: tuple[ServerInstanceSpecV1, ...] = ()
        self._match_ids: dict[str, str] = {}

    @property
    def processes(self) -> dict[str, Any]:
        return dict(self._processes)

    @property
    def match_ids(self) -> dict[str, str]:
        return dict(self._match_ids)

    def start(
        self,
        specs: Sequence[ServerInstanceSpecV1],
        *,
        dry_run: bool = False,
    ) -> tuple[ServerInstanceSpecV1, ...]:
        selected = tuple(specs)
        if not 1 <= len(selected) <= self.max_servers:
            raise ValueError(f"server count must be between 1 and {self.max_servers}")
        self._validate_specs(selected)
        if dry_run:
            return selected
        executable = self.server_root / "srcds.exe"
        if not executable.is_file():
            raise FileNotFoundError(f"dedicated server executable not found: {executable}")
        if self._processes:
            raise RuntimeError("server farm is already running")
        self._specs = selected
        try:
            for spec in selected:
                self._start_instance(spec)
        except Exception:
            self.stop(timeout_s=0)
            raise
        return selected

    def load_get5_match(
        self,
        generation: int,
        *,
        instance_id: str | None = None,
    ) -> tuple[Path, ...]:
        specs = self._active_specs(instance_id)
        paths: list[Path] = []
        for spec in specs:
            match_id = f"slbots-{self.run_id}-g{generation:03d}-i{spec.instance_id}"
            config = render_get5_config(spec, match_id, generation=generation)
            _atomic_write_json(spec.get5_config_path, config)
            relative = spec.get5_config_path.relative_to(self.server_root / "csgo")
            console_path = relative.as_posix()
            process = self._processes[spec.instance_id]
            if process.poll() is not None:
                raise RuntimeError(f"server instance {spec.instance_id} exited before match load")
            self._send_console(process, f"get5_loadmatch {console_path}")
            self._send_console(process, "get5_forcestart")
            self._send_console(process, "bot_quota 10")
            self._send_console(process, "bot_join_after_player 0")
            self._send_console(process, "bot_auto_vacate 0")
            self._send_console(process, "bot_kick")
            for _ in range(5):
                self._send_console(process, "bot_add_t")
                self._send_console(process, "bot_add_ct")
            self._match_ids[spec.instance_id] = match_id
            paths.append(spec.get5_config_path)
        return tuple(paths)

    def set_policy_generation(self, generation: int, *, instance_id: str | None = None) -> None:
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        specs = self._active_specs(instance_id)
        for spec in specs:
            process = self._processes[spec.instance_id]
            if process.poll() is not None:
                raise RuntimeError(f"server instance {spec.instance_id} exited before policy update")
            self._send_console(process, f"sl_bots_policy_generation {generation}")

    def restart_instance(self, instance_id: str, *, timeout_s: float = 10.0) -> None:
        spec = self._spec_for_instance(instance_id)
        self._active_specs(spec.instance_id)
        self.stop_instance(spec.instance_id, timeout_s=timeout_s)
        self._start_instance(spec)

    def stop_instance(self, instance_id: str, *, timeout_s: float = 10.0) -> None:
        spec = self._spec_for_instance(instance_id)
        process = self._processes.pop(spec.instance_id, None)
        if process is None:
            raise RuntimeError(f"server instance {spec.instance_id} is not running")
        try:
            self._stop_process(process, timeout_s=timeout_s)
        finally:
            handles = self._log_handles.pop(spec.instance_id, ())
            for handle in handles:
                handle.close()
            self._match_ids.pop(spec.instance_id, None)

    def stop(self, *, timeout_s: float = 10.0) -> None:
        processes = tuple(self._processes.values())
        for process in processes:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("quit\n")
                process.stdin.flush()
        deadline = time.perf_counter() + max(0.0, timeout_s)
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.perf_counter())
            try:
                process.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, TimeoutError):
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, TimeoutError):
                    process.kill()
                    process.wait(timeout=2.0)
        for stdout, stderr in self._log_handles.values():
            stdout.close()
            stderr.close()
        self._log_handles.clear()
        self._processes.clear()
        self._specs = ()
        self._match_ids.clear()

    def _start_instance(self, spec: ServerInstanceSpecV1) -> None:
        if spec.instance_id in self._processes:
            raise RuntimeError(f"server instance {spec.instance_id} is already running")
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (spec.log_dir / "server.stdout.log").open("ab")
        stderr = (spec.log_dir / "server.stderr.log").open("ab")
        try:
            process = self._process_factory(
                self._start_command(spec),
                cwd=str(self.server_root),
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                text=True,
                bufsize=1,
                **_hidden_popen_options(),
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        self._log_handles[spec.instance_id] = (stdout, stderr)
        self._processes[spec.instance_id] = process

    @staticmethod
    def _stop_process(process: Any, *, timeout_s: float) -> None:
        if process.poll() is None and process.stdin is not None:
            process.stdin.write("quit\n")
            process.stdin.flush()
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=max(0.0, timeout_s))
        except (subprocess.TimeoutExpired, TimeoutError):
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, TimeoutError):
                process.kill()
                process.wait(timeout=2.0)

    def _spec_for_instance(self, instance_id: str) -> ServerInstanceSpecV1:
        normalized = str(instance_id)
        for spec in self._specs:
            if spec.instance_id == normalized:
                return spec
        raise ValueError(f"unknown server instance: {instance_id}")

    def _active_specs(self, instance_id: str | None) -> tuple[ServerInstanceSpecV1, ...]:
        expected = {spec.instance_id for spec in self._specs}
        if not self._specs or set(self._processes) != expected:
            raise RuntimeError("server farm must be started before operating on a match")
        if instance_id is None:
            return self._specs
        spec = self._spec_for_instance(instance_id)
        if spec.instance_id not in self._processes:
            raise RuntimeError(f"server instance {spec.instance_id} is not running")
        return (spec,)

    def _validate_specs(self, specs: Sequence[ServerInstanceSpecV1]) -> None:
        ids = [spec.instance_id for spec in specs]
        if len(set(ids)) != len(ids):
            raise ValueError("instance IDs must be unique")
        ports = [port for spec in specs for port in spec.ports]
        if len(set(ports)) != len(ports):
            raise ValueError("server ports must be unique")
        ipc_names = [spec.ipc_name for spec in specs]
        if len(set(ipc_names)) != len(ipc_names):
            raise ValueError("IPC names must be unique")
        for spec in specs:
            if not _INSTANCE_PATTERN.fullmatch(spec.instance_id):
                raise ValueError("instance_id must be a two-digit value")

    @staticmethod
    def _start_command(spec: ServerInstanceSpecV1) -> list[str]:
        return [
            "srcds.exe",
            "-game",
            "csgo",
            "-console",
            "-usercon",
            "-tickrate",
            "128",
            "-port",
            str(spec.game_port),
            "+clientport",
            str(spec.client_port),
            "+tv_port",
            str(spec.tv_port),
            "-steamport",
            str(spec.steam_port),
            "+sl_bots_instance_id",
            spec.instance_id,
            "+get5_server_id",
            f"slbots-{spec.instance_id}",
            "+map",
            "de_mirage",
        ]

    @staticmethod
    def _send_console(process: Any, command: str) -> None:
        if process.stdin is None:
            raise RuntimeError("server process stdin is not available")
        process.stdin.write(command + "\n")
        process.stdin.flush()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


__all__ = [
    "DedicatedServerFarm",
    "ServerInstanceSpecV1",
    "build_server_specs",
    "render_get5_config",
    "server_get5_path",
]
