from pathlib import Path

import pytest

from abci_job import ConfigurationError
from abci_job.experiments import (
    Experiment,
    ExperimentManifest,
    load_experiment_manifest,
)

VALID_MANIFEST = """
[[experiments]]
name = "run-a"
command = ["python", "-m", "package.train", "seed=1"]

[[experiments]]
name = "run-b"
command = ["bash", "scripts/run.sh", "two words"]
""".strip()


def write_manifest(tmp_path: Path, text: str = VALID_MANIFEST) -> Path:
    path = tmp_path / "experiments.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_experiment_manifest_returns_immutable_typed_entries(tmp_path: Path):
    manifest = load_experiment_manifest(write_manifest(tmp_path))

    assert manifest == ExperimentManifest(
        experiments=(
            Experiment(
                name="run-a",
                command=("python", "-m", "package.train", "seed=1"),
            ),
            Experiment(
                name="run-b",
                command=("bash", "scripts/run.sh", "two words"),
            ),
        )
    )


def test_load_experiment_manifest_accepts_one_entry(tmp_path: Path):
    text = '[[experiments]]\nname = "run-a"\ncommand = ["true"]'

    manifest = load_experiment_manifest(write_manifest(tmp_path, text))

    assert manifest == ExperimentManifest((Experiment("run-a", ("true",)),))


def test_load_experiment_manifest_accepts_eight_entries(tmp_path: Path):
    text = "\n\n".join(
        f'[[experiments]]\nname = "run-{index}"\ncommand = ["true"]'
        for index in range(8)
    )

    manifest = load_experiment_manifest(write_manifest(tmp_path, text))

    assert len(manifest.experiments) == 8


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "experiments"),
        ("experiments = []", "between 1 and 8"),
        ('unexpected = true\n[[experiments]]\nname = "a"\ncommand = ["true"]', "unexpected"),
        ('[experiments]\nname = "a"\ncommand = ["true"]', "array of tables"),
        ('[[experiments]]\ncommand = ["true"]', "name"),
        ('[[experiments]]\nname = "a"', "command"),
        ('[[experiments]]\nname = "a"\ncommand = ["true"]\nextra = 1', "extra"),
        ('[[experiments]]\nname = 1\ncommand = ["true"]', "name"),
        ('[[experiments]]\nname = "a"\ncommand = "true"', "command"),
        ('[[experiments]]\nname = "bad/name"\ncommand = ["true"]', "scheduler-safe"),
        ('[[experiments]]\nname = "a"\ncommand = []', "at least one"),
        ('[[experiments]]\nname = "a"\ncommand = [1]', "strings"),
        ('[[experiments]]\nname = "a"\ncommand = ["line\\nfeed"]', "newline"),
        ('[[experiments]]\nname = "a"\ncommand = ["carriage\\rreturn"]', "newline"),
    ],
)
def test_load_experiment_manifest_rejects_invalid_schema(
    tmp_path: Path, text: str, message: str
):
    with pytest.raises(ConfigurationError, match=message):
        load_experiment_manifest(write_manifest(tmp_path, text))


def test_load_experiment_manifest_rejects_duplicate_names(tmp_path: Path):
    text = """
[[experiments]]
name = "same"
command = ["true"]

[[experiments]]
name = "same"
command = ["false"]
""".strip()

    with pytest.raises(ConfigurationError, match="unique"):
        load_experiment_manifest(write_manifest(tmp_path, text))


def test_load_experiment_manifest_rejects_nine_entries(tmp_path: Path):
    text = "\n\n".join(
        f'[[experiments]]\nname = "run-{index}"\ncommand = ["true"]'
        for index in range(9)
    )

    with pytest.raises(ConfigurationError, match="between 1 and 8"):
        load_experiment_manifest(write_manifest(tmp_path, text))


@pytest.mark.parametrize("filename", ["missing.toml", "invalid.toml"])
def test_load_experiment_manifest_wraps_read_errors(
    tmp_path: Path, filename: str
):
    path = tmp_path / filename
    if filename == "invalid.toml":
        path.write_text("experiments = =", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=filename):
        load_experiment_manifest(path)
