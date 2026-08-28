from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SCHEDULER_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WALLTIME_PATTERN = re.compile(r"^\d{1,3}:[0-5]\d:[0-5]\d$")
_REQUIRED_CONFIG_KEYS = {"group", "queue", "walltime", "workdir"}
_OPTIONAL_CONFIG_KEYS = {"join_output", "setup_commands", "monitor"}
_MONITOR_KEYS = {"enabled", "interval_seconds", "commands"}


class ABCIJobError(Exception):
    """Base exception for job submission errors."""


class ConfigurationError(ABCIJobError):
    """Raised when a configuration file is invalid."""


class SubmissionError(ABCIJobError):
    """Raised when scheduler submission fails."""


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool
    interval_seconds: int
    commands: tuple[str, ...]


@dataclass(frozen=True)
class ABCIConfig:
    group: str
    queue: str
    walltime: str
    workdir: Path
    join_output: bool
    setup_commands: tuple[str, ...]
    monitor: MonitorConfig


def load_config(path: str | Path) -> ABCIConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"configuration {config_path}: {error}") from error

    _validate_top_level_keys(data)
    group = _validate_scheduler_value(data, "group")
    queue = _validate_scheduler_value(data, "queue")
    walltime = _validate_walltime(data)
    workdir = _validate_workdir(data)
    join_output = _validate_bool(data.get("join_output", True), "join_output")
    setup_commands = _validate_commands(
        data.get("setup_commands", []), "setup_commands"
    )
    monitor = _validate_monitor(data.get("monitor"))

    return ABCIConfig(
        group=group,
        queue=queue,
        walltime=walltime,
        workdir=workdir,
        join_output=join_output,
        setup_commands=setup_commands,
        monitor=monitor,
    )


def validate_job_name(name: str) -> str:
    if not isinstance(name, str) or not _JOB_NAME_PATTERN.fullmatch(name):
        raise ConfigurationError("job name is not scheduler-safe")
    return name


def _validate_top_level_keys(data: dict[str, object]) -> None:
    missing = _REQUIRED_CONFIG_KEYS - data.keys()
    if missing:
        raise ConfigurationError(f"missing required configuration key: {min(missing)}")

    unknown = data.keys() - _REQUIRED_CONFIG_KEYS - _OPTIONAL_CONFIG_KEYS
    if unknown:
        raise ConfigurationError(f"unknown configuration key: {min(unknown)}")


def _validate_scheduler_value(data: dict[str, object], field: str) -> str:
    value = _validate_string(data[field], field)
    if not _SCHEDULER_VALUE_PATTERN.fullmatch(value):
        raise ConfigurationError(f"{field} must be scheduler-safe")
    return value


def _validate_walltime(data: dict[str, object]) -> str:
    walltime = _validate_string(data["walltime"], "walltime")
    if not _WALLTIME_PATTERN.fullmatch(walltime):
        raise ConfigurationError("walltime must use HHH:MM:SS")

    hours, minutes, seconds = (int(component) for component in walltime.split(":"))
    if hours == minutes == seconds == 0:
        raise ConfigurationError("walltime must be non-zero")
    return walltime


def _validate_workdir(data: dict[str, object]) -> Path:
    workdir = Path(_validate_string(data["workdir"], "workdir"))
    if not workdir.is_absolute():
        raise ConfigurationError("workdir must be absolute")
    return workdir


def _validate_monitor(value: object | None) -> MonitorConfig:
    if value is None:
        return MonitorConfig(enabled=False, interval_seconds=0, commands=())
    if not isinstance(value, dict):
        raise ConfigurationError("monitor must be a table")

    unknown = value.keys() - _MONITOR_KEYS
    if unknown:
        raise ConfigurationError(f"unknown monitor key: {min(unknown)}")

    enabled = _validate_bool(value.get("enabled", False), "monitor.enabled")
    interval_seconds = 0
    if "interval_seconds" in value:
        interval_seconds = _validate_monitor_interval(value["interval_seconds"])
    commands = _validate_commands(value.get("commands", []), "monitor.commands")
    if enabled and interval_seconds <= 0:
        raise ConfigurationError("monitor.interval_seconds must be positive")
    if enabled and not commands:
        raise ConfigurationError("monitor.commands cannot be empty when enabled")
    return MonitorConfig(enabled, interval_seconds, commands)


def _validate_monitor_interval(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("monitor.interval_seconds must be an integer")
    if value <= 0:
        raise ConfigurationError("monitor.interval_seconds must be positive")
    return value


def _validate_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a boolean")
    return value


def _validate_commands(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{field} must be a list")
    return tuple(_validate_string(command, field) for command in value)


def _validate_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field} must be a string")
    if "\n" in value or "\r" in value:
        raise ConfigurationError(f"{field} cannot contain a newline")
    return value
