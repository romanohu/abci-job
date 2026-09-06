from __future__ import annotations

import re
import shlex
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SCHEDULER_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WALLTIME_PATTERN = re.compile(r"^\d{1,3}:[0-5]\d:[0-5]\d$")
_ENVIRONMENT_KEYS = {"group", "workdir", "setup_commands", "monitor"}
_JOB_KEYS = {"name", "queue", "walltime", "command", "experiments", "rtype"}
_EXPERIMENT_KEYS = {"name", "command"}
_MONITOR_KEYS = {"enabled", "interval_seconds", "commands"}
_RESERVATION_RTYPES = {"rt_HC", "rt_HF", "rt_HG"}


class ABCIJobError(Exception):
    """Base exception for ABCI job errors."""


class ConfigurationError(ABCIJobError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool = False
    interval_seconds: int = 0
    commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_bool(self.enabled, "monitor.enabled")
        if isinstance(self.interval_seconds, bool) or not isinstance(
            self.interval_seconds, int
        ):
            raise ConfigurationError("monitor.interval_seconds must be an integer")
        if self.interval_seconds < 0 or (self.enabled and self.interval_seconds == 0):
            raise ConfigurationError("monitor.interval_seconds must be positive")
        _require_string_tuple(self.commands, "monitor.commands")
        if self.enabled and not self.commands:
            raise ConfigurationError("monitor.commands cannot be empty when enabled")


@dataclass(frozen=True)
class EnvironmentConfig:
    group: str
    workdir: Path
    setup_commands: tuple[str, ...] = ()
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self) -> None:
        _require_scheduler_value(self.group, "group")
        if not isinstance(self.workdir, Path):
            raise ConfigurationError("workdir must be a Path")
        if not self.workdir.is_absolute():
            raise ConfigurationError("workdir must be absolute")
        _reject_controls(str(self.workdir), "workdir")
        _require_string_tuple(self.setup_commands, "setup_commands")
        if not isinstance(self.monitor, MonitorConfig):
            raise ConfigurationError("monitor must be a MonitorConfig")


@dataclass(frozen=True)
class Experiment:
    name: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            validate_job_name(self.name)
        except ConfigurationError as error:
            raise ConfigurationError("experiment name is not scheduler-safe") from error
        if not isinstance(self.command, tuple):
            raise ConfigurationError("experiment command must be a tuple")
        quote_command(self.command)


@dataclass(frozen=True)
class JobConfig:
    name: str
    queue: str
    walltime: str
    experiments: tuple[Experiment, ...]
    rtype: str | None = None

    def __post_init__(self) -> None:
        validate_job_name(self.name)
        _require_scheduler_value(self.queue, "queue")
        _require_walltime(self.walltime)
        if not isinstance(self.experiments, tuple):
            raise ConfigurationError("experiments must be a tuple")
        if not 1 <= len(self.experiments) <= 8:
            raise ConfigurationError("experiments must contain between 1 and 8 entries")
        if any(not isinstance(item, Experiment) for item in self.experiments):
            raise ConfigurationError("experiments must contain Experiment values")
        names = [item.name for item in self.experiments]
        if len(names) != len(set(names)):
            raise ConfigurationError("experiment names must be unique")

        if self.rtype is not None:
            _require_scheduler_value(self.rtype, "rtype")
        if self.queue.startswith("rt_"):
            if self.rtype is not None:
                raise ConfigurationError("rtype must be absent for an ordinary queue")
        elif self.rtype is None:
            raise ConfigurationError("reservation queue requires rtype")
        elif self.rtype not in _RESERVATION_RTYPES:
            raise ConfigurationError("reservation queue rtype is not supported")
        if len(self.experiments) > 1 and (self.rtype or self.queue) != "rt_HF":
            raise ConfigurationError("multiple experiments require rt_HF")


def load_environment(path: str | Path) -> EnvironmentConfig:
    config_path = Path(path)
    try:
        data = _read_toml(config_path)
        _validate_keys(data, {"group", "workdir"}, _ENVIRONMENT_KEYS, "environment")
        group = _require_string(data["group"], "group")
        workdir = Path(_require_string(data["workdir"], "workdir"))
        setup_commands = _load_string_list(
            data.get("setup_commands", []), "setup_commands"
        )
        monitor = _load_monitor(data.get("monitor"))
        return EnvironmentConfig(group, workdir, setup_commands, monitor)
    except ConfigurationError as error:
        raise ConfigurationError(
            f"environment configuration {config_path}: {error}"
        ) from error


def load_job(
    path: str | Path, command: Sequence[str] | None = None
) -> JobConfig:
    config_path = Path(path)
    try:
        data = _read_toml(config_path)
        _validate_keys(data, {"name", "queue", "walltime"}, _JOB_KEYS, "job")

        configured_command = None
        if "command" in data:
            configured_command = _load_command(data["command"], "command")

        configured_experiments = None
        if "experiments" in data:
            entries = data["experiments"]
            if not isinstance(entries, list):
                raise ConfigurationError("experiments must be an array of tables")
            configured_experiments = tuple(
                _load_experiment(entry, index) for index, entry in enumerate(entries)
            )

        if configured_command is not None and configured_experiments is not None:
            raise ConfigurationError("command and experiments are mutually exclusive")

        cli_command = None
        if command is not None:
            cli_command = _validate_command_sequence(command, "CLI command")
        if cli_command is not None and configured_experiments is not None:
            raise ConfigurationError("CLI command with experiments is ambiguous")

        selected_command = cli_command or configured_command
        if selected_command is not None:
            experiments = (Experiment("main", selected_command),)
        elif configured_experiments is not None:
            experiments = configured_experiments
        else:
            raise ConfigurationError("job requires command or experiments")

        rtype = data.get("rtype")
        if rtype is not None:
            rtype = _require_string(rtype, "rtype")
        return JobConfig(
            name=_require_string(data["name"], "name"),
            queue=_require_string(data["queue"], "queue"),
            walltime=_require_string(data["walltime"], "walltime"),
            experiments=experiments,
            rtype=rtype,
        )
    except ConfigurationError as error:
        raise ConfigurationError(f"job configuration {config_path}: {error}") from error


def validate_job_name(name: str) -> str:
    if not isinstance(name, str) or not _JOB_NAME_PATTERN.fullmatch(name):
        raise ConfigurationError("job name is not scheduler-safe")
    return name


def quote_command(command: Sequence[str]) -> str:
    if isinstance(command, (str, bytes)) or not command:
        raise ConfigurationError("command must contain at least one argument")
    if any(not isinstance(argument, str) for argument in command):
        raise ConfigurationError("command arguments must be strings")
    for argument in command:
        _reject_controls(argument, "command arguments")
    if command[0] == "":
        raise ConfigurationError("command executable cannot be empty")
    return " ".join(shlex.quote(argument) for argument in command)


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(str(error)) from error


def _validate_keys(
    data: dict[str, object], required: set[str], allowed: set[str], context: str
) -> None:
    missing = required - data.keys()
    if missing:
        raise ConfigurationError(f"missing required {context} key: {min(missing)}")
    unknown = data.keys() - allowed
    if unknown:
        raise ConfigurationError(f"unknown {context} key: {min(unknown)}")


def _load_monitor(value: object | None) -> MonitorConfig:
    if value is None:
        return MonitorConfig()
    if not isinstance(value, dict):
        raise ConfigurationError("monitor must be a table")
    unknown = value.keys() - _MONITOR_KEYS
    if unknown:
        raise ConfigurationError(f"unknown monitor key: {min(unknown)}")
    enabled = _require_bool(value.get("enabled", False), "monitor.enabled")
    interval = value.get("interval_seconds", 0)
    if "interval_seconds" in value and (
        isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0
    ):
        raise ConfigurationError("monitor.interval_seconds must be a positive integer")
    commands = _load_string_list(value.get("commands", []), "monitor.commands")
    return MonitorConfig(enabled, interval, commands)


def _load_experiment(value: object, index: int) -> Experiment:
    context = f"experiments[{index}]"
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be a table")
    _validate_keys(value, _EXPERIMENT_KEYS, _EXPERIMENT_KEYS, context)
    try:
        return Experiment(
            _require_string(value["name"], f"{context}.name"),
            _load_command(value["command"], f"{context}.command"),
        )
    except ConfigurationError as error:
        raise ConfigurationError(f"{context}: {error}") from error


def _load_command(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{field_name} must be an array")
    return _validate_command_sequence(value, field_name)


def _validate_command_sequence(
    value: Sequence[str], field_name: str
) -> tuple[str, ...]:
    try:
        quote_command(value)
    except ConfigurationError as error:
        raise ConfigurationError(f"{field_name}: {error}") from error
    return tuple(value)


def _load_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{field_name} must be a list")
    return tuple(_require_string(item, field_name) for item in value)


def _require_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ConfigurationError(f"{field_name} must be a tuple")
    for item in value:
        _require_string(item, field_name)


def _require_scheduler_value(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    if not _SCHEDULER_VALUE_PATTERN.fullmatch(text):
        raise ConfigurationError(f"{field_name} must be scheduler-safe")
    return text


def _require_walltime(value: object) -> str:
    walltime = _require_string(value, "walltime")
    if not _WALLTIME_PATTERN.fullmatch(walltime):
        raise ConfigurationError("walltime must use HHH:MM:SS")
    if all(int(part) == 0 for part in walltime.split(":")):
        raise ConfigurationError("walltime must be non-zero")
    return walltime


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a boolean")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} must be a string")
    _reject_controls(value, field_name)
    return value


def _reject_controls(value: str, field_name: str) -> None:
    if "\n" in value or "\r" in value:
        raise ConfigurationError(f"{field_name} cannot contain a newline")
    if "\x00" in value:
        raise ConfigurationError(f"{field_name} cannot contain NUL")
