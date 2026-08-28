import os
import subprocess
from pathlib import Path

import pytest

from submit import main, parse_args

CONFIG_TOML = """
group = "example-group"
queue = "rt_HG"
walltime = "01:00:00"
workdir = "/tmp"
""".strip()


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "abci.toml"
    config_path.write_text(CONFIG_TOML, encoding="utf-8")
    return config_path


def test_parse_args_preserves_every_command_token_after_separator():
    args = parse_args(
        [
            "--config",
            "configs/abci_default.toml",
            "--name",
            "example-job",
            "--dry-run",
            "--",
            "python",
            "-m",
            "package.train",
            "--label",
            "run one",
        ]
    )

    assert args.config == Path("configs/abci_default.toml")
    assert args.name == "example-job"
    assert args.dry_run is True
    assert args.command == ["python", "-m", "package.train", "--label", "run one"]


def test_parse_args_supports_print_script():
    args = parse_args(
        [
            "--config",
            "config.toml",
            "--name",
            "example-job",
            "--print-script",
            "--",
            "true",
        ]
    )

    assert args.print_script is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--name", "example-job", "--", "true"],
        ["--config", "config.toml", "--", "true"],
        ["--config", "config.toml", "--name", "example-job", "true"],
        ["--config", "config.toml", "--name", "example-job", "--"],
    ],
)
def test_parse_args_requires_helper_options_separator_and_command(argv: list[str]):
    with pytest.raises(SystemExit, match="2"):
        parse_args(argv)


def test_parse_args_keeps_helper_options_after_separator_as_workload_arguments():
    args = parse_args(
        [
            "--config",
            "config.toml",
            "--name",
            "example-job",
            "--",
            "python",
            "--dry-run",
            "--print-script",
        ]
    )

    assert args.dry_run is False
    assert args.print_script is False
    assert args.command == ["python", "--dry-run", "--print-script"]


def test_main_dry_run_writes_executable_script_without_submitting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    jobs_dir = tmp_path / "jobs"

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("dry-run must not invoke the scheduler")

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--name",
            "example-job",
            "--dry-run",
            "--",
            "python",
            "-m",
            "package.train",
        ],
        submit_runner=runner,
        jobs_dir=jobs_dir,
    )

    job_path = jobs_dir / "example-job.sh"
    captured = capsys.readouterr()
    assert result == 0
    assert job_path.exists()
    assert os.access(job_path, os.X_OK)
    assert f"Generated job script: {job_path}" in captured.out
    assert captured.err == ""


def test_main_submits_once_and_reports_generated_path_and_job_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="12345.pbs1\n")

    jobs_dir = tmp_path / "jobs"
    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--name",
            "example-job",
            "--",
            "true",
        ],
        submit_runner=runner,
        jobs_dir=jobs_dir,
    )

    job_path = jobs_dir / "example-job.sh"
    captured = capsys.readouterr()
    assert result == 0
    assert calls == [
        (
            (["qsub", str(job_path)],),
            {"check": True, "capture_output": True, "text": True},
        )
    ]
    assert f"Generated job script: {job_path}" in captured.out
    assert "Submitted job: 12345.pbs1" in captured.out
    assert captured.err == ""


def test_main_print_script_outputs_rendered_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--name",
            "example-job",
            "--dry-run",
            "--print-script",
            "--",
            "python",
            "-m",
            "package.train",
            "--output",
            "results/run one",
        ],
        jobs_dir=tmp_path / "jobs",
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "#!/bin/bash -l" in captured.out
    assert "python -m package.train --output 'results/run one'" in captured.out


def test_main_reports_invalid_configuration_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    result = main(
        [
            "--config",
            str(tmp_path / "missing.toml"),
            "--name",
            "example-job",
            "--dry-run",
            "--",
            "true",
        ],
        jobs_dir=tmp_path / "jobs",
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err.startswith("error: configuration")
    assert "Traceback" not in captured.err


def test_main_reports_scheduler_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError()

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--name",
            "example-job",
            "--",
            "true",
        ],
        submit_runner=runner,
        jobs_dir=tmp_path / "jobs",
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "error: qsub executable was not found\n"
    assert "Traceback" not in captured.err


def test_main_reports_multiline_scheduler_error_on_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            1, ["qsub"], stderr="first line\nsecond line"
        )

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--name",
            "example-job",
            "--",
            "true",
        ],
        submit_runner=runner,
        jobs_dir=tmp_path / "jobs",
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == "error: scheduler rejected the job: first line second line\n"


def test_main_reports_filesystem_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    jobs_path = tmp_path / "not-a-directory"
    jobs_path.write_text("file", encoding="utf-8")

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--name",
            "example-job",
            "--dry-run",
            "--",
            "true",
        ],
        jobs_dir=jobs_path,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
