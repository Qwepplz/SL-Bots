"""Windows-safe srcds launcher with a hidden, real console and CONIN$ input."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Any, Callable, Mapping, Sequence


def resolve_srcds_executable(server_root: str | Path) -> Path:
    """Resolve srcds.exe from the explicit server root, never from the CWD."""

    root = Path(server_root).expanduser().resolve(strict=False)
    if root.name.casefold() == "srcds.exe":
        executable = root
    else:
        executable = root / "srcds.exe"
    executable = executable.resolve(strict=False)
    if executable.name.casefold() != "srcds.exe" or not executable.is_file():
        raise FileNotFoundError(f"dedicated server executable not found: {executable}")
    return executable


def build_srcds_command(
    executable: str | Path,
    *,
    game_port: int,
    rcon_password: str,
    client_port: int | None = None,
    tv_port: int | None = None,
    steam_port: int | None = None,
    instance_id: str | None = None,
    map_name: str = "de_mirage",
) -> list[str]:
    if not rcon_password:
        raise ValueError("RCON password is required")
    command = [
        str(Path(executable).resolve(strict=False)),
        "-game",
        "csgo",
        "-console",
        "-usercon",
        "-tickrate",
        "128",
        "-port",
        str(game_port),
        "-ip",
        "127.0.0.1",
        "+rcon_password",
        rcon_password,
        "+map",
        map_name,
    ]
    for name, value in (
        ("+clientport", client_port),
        ("+tv_port", tv_port),
        ("-steamport", steam_port),
    ):
        if value is not None:
            command.extend((name, str(value)))
    if instance_id is not None:
        command.extend(("+sl_bots_instance_id", str(instance_id)))
    return command


def open_console_input(*, platform_name: str | None = None) -> Any:
    """Open the parent's console input on Windows; never return a pipe."""

    resolved_platform = os.name if platform_name is None else platform_name
    if resolved_platform != "nt":
        return subprocess.DEVNULL
    return open("CONIN$", "rb", buffering=0)


def build_hidden_console_popen_kwargs(
    *,
    platform_name: str | None = None,
    console_input: Any = None,
    stdout: Any,
    stderr: Any,
) -> dict[str, Any]:
    """Build Popen arguments without ``stdin=PIPE`` or ``CREATE_NO_WINDOW``."""

    resolved_platform = os.name if platform_name is None else platform_name
    if console_input is None:
        console_input = open_console_input(platform_name=resolved_platform)
    kwargs: dict[str, Any] = {
        "stdin": console_input,
        "stdout": stdout,
        "stderr": stderr,
    }
    if resolved_platform != "nt":
        return kwargs
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW | subprocess.STARTF_USESTDHANDLES
    startupinfo.wShowWindow = subprocess.SW_HIDE
    kwargs.update(
        {
            "startupinfo": startupinfo,
            "creationflags": subprocess.CREATE_NEW_CONSOLE,
            "close_fds": False,
        }
    )
    return kwargs


def launch_srcds(
    server_root: str | Path,
    arguments: Sequence[str],
    *,
    stdout: Any,
    stderr: Any,
    process_factory: Callable[..., Any] = subprocess.Popen,
) -> Any:
    """Launch one srcds process using an explicit absolute executable path."""

    executable = resolve_srcds_executable(server_root)
    command = [str(executable), *(str(argument) for argument in arguments)]
    console_input = open_console_input()
    kwargs = build_hidden_console_popen_kwargs(
        console_input=console_input,
        stdout=stdout,
        stderr=stderr,
    )
    try:
        return process_factory(command, cwd=str(executable.parent), **kwargs)
    finally:
        if hasattr(console_input, "close"):
            console_input.close()


__all__ = [
    "build_hidden_console_popen_kwargs",
    "build_srcds_command",
    "launch_srcds",
    "open_console_input",
    "resolve_srcds_executable",
]
