from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .submitter import ConfigurationError, quote_command, validate_job_name

_MAX_EXPERIMENTS = 8
_TOP_LEVEL_KEYS = {"experiments"}
_EXPERIMENT_KEYS = {"name", "command"}


@dataclass(frozen=True)
class Experiment:
    name: str
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            validate_job_name(self.name)
        except ConfigurationError as error:
            raise ConfigurationError(
                "experiment name is not scheduler-safe"
            ) from error
        if not isinstance(self.command, tuple):
            raise ConfigurationError("experiment command must be a tuple")
        quote_command(self.command)


@dataclass(frozen=True)
class ExperimentManifest:
    experiments: tuple[Experiment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.experiments, tuple):
            raise ConfigurationError("experiments must be a tuple")
        if not 1 <= len(self.experiments) <= _MAX_EXPERIMENTS:
            raise ConfigurationError("experiments must contain between 1 and 8 entries")
        if any(not isinstance(experiment, Experiment) for experiment in self.experiments):
            raise ConfigurationError("experiments must contain Experiment values")
        names = [experiment.name for experiment in self.experiments]
        if len(names) != len(set(names)):
            raise ConfigurationError("experiment names must be unique")


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    manifest_path = Path(path)
    try:
        with manifest_path.open("rb") as manifest_file:
            data = tomllib.load(manifest_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(
            f"experiment manifest {manifest_path}: {error}"
        ) from error

    missing = _TOP_LEVEL_KEYS - data.keys()
    if missing:
        raise ConfigurationError("missing experiment manifest key: experiments")
    unknown = data.keys() - _TOP_LEVEL_KEYS
    if unknown:
        raise ConfigurationError(f"unknown experiment manifest key: {min(unknown)}")

    entries = data["experiments"]
    if not isinstance(entries, list):
        raise ConfigurationError("experiments must be an array of tables")
    if not 1 <= len(entries) <= _MAX_EXPERIMENTS:
        raise ConfigurationError("experiments must contain between 1 and 8 entries")

    experiments = tuple(
        _parse_experiment(entry, index) for index, entry in enumerate(entries)
    )
    return ExperimentManifest(experiments=experiments)


def _parse_experiment(value: object, index: int) -> Experiment:
    field = f"experiments[{index}]"
    if not isinstance(value, dict):
        raise ConfigurationError(f"{field} must be a table")

    missing = _EXPERIMENT_KEYS - value.keys()
    if missing:
        raise ConfigurationError(f"missing {field} key: {min(missing)}")
    unknown = value.keys() - _EXPERIMENT_KEYS
    if unknown:
        raise ConfigurationError(f"unknown {field} key: {min(unknown)}")

    name = value["name"]
    if not isinstance(name, str):
        raise ConfigurationError(f"{field}.name must be a string")
    command = value["command"]
    if not isinstance(command, list):
        raise ConfigurationError(f"{field}.command must be an array")

    try:
        return Experiment(name=name, command=tuple(command))
    except ConfigurationError as error:
        raise ConfigurationError(f"{field}: {error}") from error
