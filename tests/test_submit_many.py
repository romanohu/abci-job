import os
import subprocess
import sys
from pathlib import Path

import pytest

from submit_many import main, parse_args

CONFIG_TOML = """
group = "example-group"
queue = "rt_HF"
walltime = "01:00:00"
workdir = "WORKDIR"
""".strip()

MANIFEST_TOML = """
[[experiments]]
name = "run-a"
command = ["python", "-c", "print('a')"]

[[experiments]]
name = "run-b"
command = ["python", "-c", "print('b')"]
""".strip()


def write_config(tmp_path: Path, *, queue: str = "rt_HF") -> Path:
    path = tmp_path / "abci.toml"
    path.write_text(
        CONFIG_TOML.replace("rt_HF", queue).replace(
            "WORKDIR", str(tmp_path / "workdir")
        ),
        encoding="utf-8",
    )
    return path


def write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "experiments.toml"
    path.write_text(MANIFEST_TOML, encoding="utf-8")
    return path


def test_parse_args_accepts_multi_experiment_options():
    args = parse_args(
        [
            "--config",
            "config.toml",
            "--experiments",
            "runs.toml",
            "--name",
            "batch-a",
            "--dry-run",
            "--print-script",
        ]
    )

    assert args.config == Path("config.toml")
    assert args.experiments == Path("runs.toml")
    assert args.name == "batch-a"
    assert args.dry_run is True
    assert args.print_script is True


def test_parse_args_rejects_single_command_workload_boundary():
    with pytest.raises(SystemExit) as error:
        parse_args(
            [
                "--config",
                "config.toml",
                "--experiments",
                "runs.toml",
                "--name",
                "batch-a",
                "--",
                "true",
            ]
        )

    assert error.value.code == 2


def test_cli_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "submit_many.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--experiments" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--experiments", "runs.toml", "--name", "batch-a"],
        ["--config", "config.toml", "--name", "batch-a"],
        ["--config", "config.toml", "--experiments", "runs.toml"],
    ],
)
def test_parse_args_requires_config_manifest_and_name(argv: list[str]):
    with pytest.raises(SystemExit, match="2"):
        parse_args(argv)


def test_main_dry_run_writes_script_without_scheduler_or_workdir_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    workdir = tmp_path / "workdir"

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("dry-run must not invoke qsub")

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--experiments",
            str(write_manifest(tmp_path)),
            "--name",
            "batch-a",
            "--dry-run",
            "--print-script",
        ],
        submit_runner=runner,
        jobs_dir=tmp_path / "jobs",
    )

    captured = capsys.readouterr()
    job_path = tmp_path / "jobs" / "batch-a.sh"
    assert result == 0
    assert job_path.exists()
    assert os.access(job_path, os.X_OK)
    assert "#PBS -q rt_HF" in captured.out
    assert "CUDA_VISIBLE_DEVICES=0" in captured.out
    assert not workdir.exists()
    assert captured.err == ""


def test_main_submits_exactly_once_and_prepares_pbs_output_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert (tmp_path / "workdir" / "logs").is_dir()
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="12345.pbs1\n")

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--experiments",
            str(write_manifest(tmp_path)),
            "--name",
            "batch-a",
        ],
        submit_runner=runner,
        jobs_dir=tmp_path / "jobs",
    )

    job_path = tmp_path / "jobs" / "batch-a.sh"
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


def test_main_rejects_non_full_node_queue_before_writing_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    jobs_dir = tmp_path / "jobs"

    result = main(
        [
            "--config",
            str(write_config(tmp_path, queue="rt_HG")),
            "--experiments",
            str(write_manifest(tmp_path)),
            "--name",
            "batch-a",
            "--dry-run",
        ],
        jobs_dir=jobs_dir,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert not jobs_dir.exists()
    assert captured.out == ""
    assert captured.err == "error: multi-experiment jobs require queue rt_HF\n"


def test_main_reports_manifest_error_on_one_line_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    invalid_manifest = tmp_path / "invalid.toml"
    invalid_manifest.write_text(
        '[[experiments]]\nname = "bad/name"\ncommand = ["true"]',
        encoding="utf-8",
    )

    result = main(
        [
            "--config",
            str(write_config(tmp_path)),
            "--experiments",
            str(invalid_manifest),
            "--name",
            "batch-a",
            "--dry-run",
        ],
        jobs_dir=tmp_path / "jobs",
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "scheduler-safe" in captured.err
    assert "Traceback" not in captured.err
