from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

_JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SCHEDULER_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WALLTIME_PATTERN = re.compile(r"^\d{1,3}:[0-5]\d:[0-5]\d$")
_JOB_ID_PATTERN = re.compile(r"^\d+(?:\.[A-Za-z0-9._-]+)?$")
_REQUIRED_CONFIG_KEYS = {"group", "queue", "walltime", "workdir"}
_OPTIONAL_CONFIG_KEYS = {"output_path", "join_output", "setup_commands", "monitor"}
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
    output_path: Path | None
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
    configured_output = data.get("output_path")
    output_path = (
        None
        if configured_output is None
        else Path(_validate_output_path_string(configured_output))
    )
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
        output_path=output_path,
        join_output=join_output,
        setup_commands=setup_commands,
        monitor=monitor,
    )


def validate_job_name(name: str) -> str:
    if not isinstance(name, str) or not _JOB_NAME_PATTERN.fullmatch(name):
        raise ConfigurationError("job name is not scheduler-safe")
    return name


def resolve_output_path(config: ABCIConfig, job_name: str) -> Path:
    if not isinstance(config, ABCIConfig):
        raise ConfigurationError("config must be an ABCIConfig")

    workdir = _validate_workdir({"workdir": str(config.workdir)})
    validated_name = validate_job_name(job_name)
    if config.output_path is None:
        return workdir / "logs" / f"{validated_name}.log"
    if not isinstance(config.output_path, Path):
        raise ConfigurationError("output_path must be a Path or None")

    configured = Path(_validate_output_path_string(str(config.output_path)))
    if configured.is_absolute():
        return configured

    normalized_workdir = Path(os.path.abspath(workdir))
    resolved = Path(os.path.abspath(normalized_workdir / configured))
    if not resolved.is_relative_to(normalized_workdir):
        raise ConfigurationError("output_path resolves outside workdir")
    return resolved


def render_job_script(
    config: ABCIConfig,
    job_name: str,
    command: Sequence[str],
    *,
    template_path: str | Path | None = None,
) -> str:
    group, queue, walltime, workdir, join_output, setup_commands, monitor = (
        _validate_render_config(config)
    )
    output_path = resolve_output_path(config, job_name)
    template = _load_template(template_path)

    return template.render(
        group=group,
        queue=queue,
        walltime=walltime,
        job_name=validate_job_name(job_name),
        join_output=join_output,
        output_path=shlex.quote(str(output_path)),
        workdir=shlex.quote(str(workdir)),
        setup_commands=setup_commands,
        monitor_enabled=monitor.enabled,
        monitor_interval=monitor.interval_seconds,
        monitor_commands=monitor.commands,
        command=_quote_command(command),
    )


def write_job_script(content: str, job_name: str, *, jobs_dir: str | Path) -> Path:
    validated_name = validate_job_name(job_name)
    destination_dir = Path(jobs_dir)
    destination = destination_dir / f"{validated_name}.sh"
    temporary_path: Path | None = None

    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination_dir, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o755)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return destination


def submit_job(
    job_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    try:
        result = runner(
            ["qsub", str(job_path)], check=True, capture_output=True, text=True
        )
    except FileNotFoundError as error:
        raise SubmissionError("qsub executable was not found") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        message = "scheduler rejected the job"
        if stderr:
            message = f"{message}: {stderr}"
        raise SubmissionError(message) from error

    job_id = result.stdout.strip()
    if not _JOB_ID_PATTERN.fullmatch(job_id):
        raise SubmissionError("scheduler returned an invalid job identifier")
    return job_id


def _validate_render_config(
    config: ABCIConfig,
) -> tuple[str, str, str, Path, bool, tuple[str, ...], MonitorConfig]:
    if not isinstance(config, ABCIConfig):
        raise ConfigurationError("config must be an ABCIConfig")
    if not isinstance(config.monitor, MonitorConfig):
        raise ConfigurationError("monitor must be a MonitorConfig")

    group = _validate_scheduler_value({"group": config.group}, "group")
    queue = _validate_scheduler_value({"queue": config.queue}, "queue")
    walltime = _validate_walltime({"walltime": config.walltime})
    workdir = _validate_workdir({"workdir": str(config.workdir)})
    join_output = _validate_bool(config.join_output, "join_output")
    setup_commands = _validate_render_commands(config.setup_commands, "setup_commands")
    monitor_data: dict[str, object] = {
        "enabled": config.monitor.enabled,
        "commands": list(config.monitor.commands),
    }
    if config.monitor.interval_seconds != 0:
        monitor_data["interval_seconds"] = config.monitor.interval_seconds
    monitor = _validate_monitor(monitor_data)
    return group, queue, walltime, workdir, join_output, setup_commands, monitor


def _validate_render_commands(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ConfigurationError(f"{field} must be a tuple")
    return tuple(_validate_string(command, field) for command in value)


def _load_template(template_path: str | Path | None):
    path = (
        Path(__file__).resolve().parents[1] / "templates" / "abci.pbs.j2"
        if template_path is None
        else Path(template_path)
    )
    environment = Environment(
        loader=FileSystemLoader(path.parent),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    try:
        return environment.get_template(path.name)
    except TemplateNotFound as error:
        raise ConfigurationError(f"template {path} was not found") from error


def _quote_command(command: Sequence[str]) -> str:
    if isinstance(command, (str, bytes)) or not command:
        raise ConfigurationError("command must contain at least one argument")
    if any(not isinstance(argument, str) for argument in command):
        raise ConfigurationError("command arguments must be strings")
    if any("\n" in argument or "\r" in argument for argument in command):
        raise ConfigurationError("command arguments cannot contain a newline")
    return " ".join(shlex.quote(argument) for argument in command)


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


def _validate_output_path_string(value: object) -> str:
    output_path = _validate_string(value, "output_path")
    if "\x00" in output_path:
        raise ConfigurationError("output_path cannot contain NUL")
    return output_path


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
