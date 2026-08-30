import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from abci_job import ABCIConfig, ConfigurationError, MonitorConfig
from abci_job.experiments import (
    Experiment,
    ExperimentManifest,
    load_experiment_manifest,
    render_multi_job_script,
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


def valid_multi_config(tmp_path: Path, **overrides: object) -> ABCIConfig:
    values: dict[str, object] = {
        "group": "example-group",
        "queue": "rt_HF",
        "walltime": "12:00:00",
        "workdir": tmp_path / "workdir",
        "output_path": None,
        "join_output": True,
        "setup_commands": (),
        "monitor": MonitorConfig(enabled=False, interval_seconds=0, commands=()),
    }
    values.update(overrides)
    return ABCIConfig(**values)  # type: ignore[arg-type]


def write_rendered_script(tmp_path: Path, script: str) -> Path:
    path = tmp_path / "job.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def run_rendered_script(script_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PBS_JOBID": "12345.pbs1"},
        timeout=10,
    )


def barrier_command(ready: Path, peer: Path) -> tuple[str, ...]:
    program = "\n".join(
        (
            "import os, pathlib, sys, time",
            f"ready = pathlib.Path({str(ready)!r})",
            f"peer = pathlib.Path({str(peer)!r})",
            "ready.write_text('ready')",
            "deadline = time.monotonic() + 3",
            "while not peer.exists():",
            "    if time.monotonic() > deadline:",
            "        sys.exit(9)",
            "    time.sleep(0.01)",
            "print(os.environ['CUDA_VISIBLE_DEVICES'])",
        )
    )
    return (sys.executable, "-c", f"exec({program!r})")


def registration_signal_setup(
    process_pid: Path,
    child_pid: Path,
    signal_number: signal.Signals,
) -> tuple[str, ...]:
    signal_name = signal_number.name.removeprefix("SIG")
    return (
        f"registration_process_pid={shlex.quote(str(process_pid))}",
        f"registration_child_pid={shlex.quote(str(child_pid))}",
        (
            "trap 'if [[ \"$BASH_COMMAND\" == "
            "\"experiment_pids+=(\\\"\\$!\\\")\" ]]; then "
            "trap - DEBUG; "
            "while [[ ! -s \"$registration_process_pid\" || "
            "! -s \"$registration_child_pid\" ]]; do sleep 0.01; done; "
            f"kill -{signal_name} \"$$\"; fi' DEBUG"
        ),
    )


def signal_after_leader_reaped_setup() -> tuple[str, ...]:
    return (
        (
            "trap 'if [[ \"$BASH_COMMAND\" == *Succeeded* ]]; then "
            "trap - DEBUG; kill -TERM \"$$\"; fi' DEBUG"
        ),
    )


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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


def test_render_multi_job_script_emits_full_node_gpu_and_log_mapping(
    tmp_path: Path,
):
    manifest = ExperimentManifest(
        experiments=(
            Experiment("run-a", ("python", "-c", "print('$HOME; *')")),
            Experiment("run-b", ("bash", "scripts/run.sh", "two words")),
        )
    )

    script = render_multi_job_script(
        valid_multi_config(tmp_path), manifest, "batch-a"
    )

    assert "#PBS -q rt_HF" in script
    assert "#PBS -l select=1" in script
    assert "#PBS -N batch-a" in script
    assert f"#PBS -o {tmp_path}/workdir/logs/batch-a.log" in script
    assert f'experiment_log_dir={tmp_path}/workdir/logs/batch-a/"$PBS_JOBID"' in script
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert "export CUDA_VISIBLE_DEVICES=1" in script
    assert shlex.join(("python", "-c", "print('$HOME; *')")) in script
    assert "bash scripts/run.sh 'two words'" in script
    assert "${experiment_log_dir}/run-a.log" in script
    assert "${experiment_log_dir}/run-b.log" in script


def test_render_multi_job_script_rejects_non_full_node_queue(tmp_path: Path):
    manifest = ExperimentManifest((Experiment("run-a", ("true",)),))

    with pytest.raises(ConfigurationError, match="rt_HF"):
        render_multi_job_script(
            valid_multi_config(tmp_path, queue="rt_HG"), manifest, "batch-a"
        )


def test_rendered_multi_job_runs_concurrently_and_writes_gpu_logs(tmp_path: Path):
    (tmp_path / "workdir").mkdir()
    ready_a = tmp_path / "ready-a"
    ready_b = tmp_path / "ready-b"
    manifest = ExperimentManifest(
        (
            Experiment("run-a", barrier_command(ready_a, ready_b)),
            Experiment("run-b", barrier_command(ready_b, ready_a)),
        )
    )
    script = render_multi_job_script(valid_multi_config(tmp_path), manifest, "batch-a")

    result = run_rendered_script(write_rendered_script(tmp_path, script))

    log_dir = tmp_path / "workdir" / "logs" / "batch-a" / "12345.pbs1"
    assert result.returncode == 0
    assert (log_dir / "run-a.log").read_text(encoding="utf-8").strip() == "0"
    assert (log_dir / "run-b.log").read_text(encoding="utf-8").strip() == "1"
    assert "All 2 experiments succeeded" in result.stdout


def test_rendered_multi_job_executes_dash_prefixed_executable_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "workdir").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    observed_gpu = tmp_path / "observed-gpu"
    executable = bin_dir / "-gpu-probe"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s' \"$CUDA_VISIBLE_DEVICES\" > {shlex.quote(str(observed_gpu))}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    manifest = ExperimentManifest(
        (Experiment(name="dash-executable", command=("-gpu-probe",)),)
    )

    script = render_multi_job_script(valid_multi_config(tmp_path), manifest, "batch-a")
    result = run_rendered_script(write_rendered_script(tmp_path, script))
    experiment_log = (
        tmp_path
        / "workdir"
        / "logs"
        / "batch-a"
        / "12345.pbs1"
        / "dash-executable.log"
    )

    assert result.returncode == 0, experiment_log.read_text(encoding="utf-8")
    assert observed_gpu.read_text(encoding="utf-8") == "0"


def test_rendered_multi_job_waits_for_peers_then_reports_aggregate_failure(
    tmp_path: Path,
):
    (tmp_path / "workdir").mkdir()
    completed = tmp_path / "peer-completed"
    manifest = ExperimentManifest(
        (
            Experiment("failure", (sys.executable, "-c", "import sys; sys.exit(7)")),
            Experiment(
                "success",
                (
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(completed)!r}).write_text('done')",
                ),
            ),
        )
    )
    script = render_multi_job_script(valid_multi_config(tmp_path), manifest, "batch-a")

    result = run_rendered_script(write_rendered_script(tmp_path, script))

    assert result.returncode == 1
    assert completed.read_text(encoding="utf-8") == "done"
    assert "Failed failure on GPU 0 with exit status 7" in result.stdout
    assert "Succeeded success on GPU 1" in result.stdout
    assert "1 of 2 experiments failed" in result.stdout


@pytest.mark.parametrize(
    ("signal_number", "expected_status"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_rendered_multi_job_cleans_experiment_group_and_monitor_on_signal(
    tmp_path: Path,
    signal_number: signal.Signals,
    expected_status: int,
):
    (tmp_path / "workdir").mkdir()
    worker = tmp_path / "worker.sh"
    worker.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "sleep 60 &\n"
        'echo "$!" > "$2"\n'
        "trap 'touch \"$3\"; exit 0' TERM INT\n"
        "while true; do wait || true; done\n",
        encoding="utf-8",
    )
    experiment_pid = tmp_path / "experiment.pid"
    experiment_child_pid = tmp_path / "experiment-child.pid"
    experiment_cleaned = tmp_path / "experiment-cleaned"
    monitor_pid = tmp_path / "monitor.pid"
    monitor_child_pid = tmp_path / "monitor-child.pid"
    monitor_cleaned = tmp_path / "monitor-cleaned"
    monitor_command = shlex.join(
        (
            "bash",
            str(worker),
            str(monitor_pid),
            str(monitor_child_pid),
            str(monitor_cleaned),
        )
    )
    config = valid_multi_config(
        tmp_path,
        monitor=MonitorConfig(True, 60, (monitor_command,)),
    )
    manifest = ExperimentManifest(
        (
            Experiment(
                "worker",
                (
                    "bash",
                    str(worker),
                    str(experiment_pid),
                    str(experiment_child_pid),
                    str(experiment_cleaned),
                ),
            ),
        )
    )
    script_path = write_rendered_script(
        tmp_path,
        render_multi_job_script(config, manifest, "signal-batch"),
    )
    process = subprocess.Popen(
        ["bash", str(script_path)],
        env={**os.environ, "PBS_JOBID": "12345.pbs1"},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tracked_pids: list[int] = []

    try:
        deadline = time.monotonic() + 5
        pid_files = (
            experiment_pid,
            experiment_child_pid,
            monitor_pid,
            monitor_child_pid,
        )
        while time.monotonic() < deadline and not all(path.exists() for path in pid_files):
            time.sleep(0.01)
        assert all(path.exists() for path in pid_files)
        tracked_pids = [int(path.read_text(encoding="utf-8")) for path in pid_files]

        cleanup_started_at = time.monotonic()
        process.send_signal(signal_number)

        assert process.wait(timeout=5) == expected_status
        assert time.monotonic() - cleanup_started_at < 0.8
        assert experiment_cleaned.exists()
        assert monitor_cleaned.exists()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
            _process_exists(pid) for pid in tracked_pids
        ):
            time.sleep(0.01)
        assert all(not _process_exists(pid) for pid in tracked_pids)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in tracked_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("signal_number", "expected_status"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_rendered_multi_job_defers_signal_until_spawned_group_is_registered(
    tmp_path: Path,
    signal_number: signal.Signals,
    expected_status: int,
):
    (tmp_path / "workdir").mkdir()
    worker = tmp_path / "worker.sh"
    worker.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "sleep 60 &\n"
        'echo "$!" > "$2"\n'
        "trap 'touch \"$3\"; exit 0' TERM INT\n"
        "while true; do wait || true; done\n",
        encoding="utf-8",
    )
    experiment_pid = tmp_path / "experiment.pid"
    experiment_child_pid = tmp_path / "experiment-child.pid"
    experiment_cleaned = tmp_path / "experiment-cleaned"
    config = valid_multi_config(
        tmp_path,
        setup_commands=registration_signal_setup(
            experiment_pid,
            experiment_child_pid,
            signal_number,
        ),
    )
    manifest = ExperimentManifest(
        (
            Experiment(
                "worker",
                (
                    "bash",
                    str(worker),
                    str(experiment_pid),
                    str(experiment_child_pid),
                    str(experiment_cleaned),
                ),
            ),
        )
    )
    script = render_multi_job_script(config, manifest, "signal-registration")
    tracked_pids: list[int] = []

    try:
        result = run_rendered_script(write_rendered_script(tmp_path, script))
        tracked_pids = [
            int(experiment_pid.read_text(encoding="utf-8")),
            int(experiment_child_pid.read_text(encoding="utf-8")),
        ]

        assert result.returncode == expected_status
        assert "unbound variable" not in result.stderr
        assert experiment_cleaned.exists()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
            _process_exists(pid) for pid in tracked_pids
        ):
            time.sleep(0.01)
        assert all(not _process_exists(pid) for pid in tracked_pids)
    finally:
        for pid in tracked_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_rendered_multi_job_bounds_stubborn_group_cleanup_and_stops_monitor(
    tmp_path: Path,
):
    (tmp_path / "workdir").mkdir()
    stubborn_worker = tmp_path / "stubborn-worker.sh"
    stubborn_worker.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "trap '' TERM INT\n"
        "(trap '' TERM INT; while true; do sleep 60; done) &\n"
        'echo "$!" > "$2"\n'
        "while true; do wait || true; done\n",
        encoding="utf-8",
    )
    monitor_worker = tmp_path / "monitor-worker.sh"
    monitor_worker.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "sleep 60 &\n"
        'echo "$!" > "$2"\n'
        "trap 'touch \"$3\"; exit 0' TERM INT\n"
        "while true; do wait || true; done\n",
        encoding="utf-8",
    )
    experiment_pid = tmp_path / "experiment.pid"
    experiment_child_pid = tmp_path / "experiment-child.pid"
    monitor_pid = tmp_path / "monitor.pid"
    monitor_child_pid = tmp_path / "monitor-child.pid"
    monitor_cleaned = tmp_path / "monitor-cleaned"
    monitor_command = shlex.join(
        (
            "bash",
            str(monitor_worker),
            str(monitor_pid),
            str(monitor_child_pid),
            str(monitor_cleaned),
        )
    )
    config = valid_multi_config(
        tmp_path,
        monitor=MonitorConfig(True, 60, (monitor_command,)),
    )
    manifest = ExperimentManifest(
        (
            Experiment(
                "stubborn",
                (
                    "bash",
                    str(stubborn_worker),
                    str(experiment_pid),
                    str(experiment_child_pid),
                ),
            ),
        )
    )
    script_path = write_rendered_script(
        tmp_path,
        render_multi_job_script(config, manifest, "stubborn-batch"),
    )
    process = subprocess.Popen(
        ["bash", str(script_path)],
        env={**os.environ, "PBS_JOBID": "12345.pbs1"},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tracked_pids: list[int] = []

    try:
        deadline = time.monotonic() + 5
        pid_files = (
            experiment_pid,
            experiment_child_pid,
            monitor_pid,
            monitor_child_pid,
        )
        while time.monotonic() < deadline and not all(path.exists() for path in pid_files):
            time.sleep(0.01)
        assert all(path.exists() for path in pid_files)
        tracked_pids = [int(path.read_text(encoding="utf-8")) for path in pid_files]

        process.send_signal(signal.SIGTERM)

        monitor_deadline = time.monotonic() + 1
        while time.monotonic() < monitor_deadline and not monitor_cleaned.exists():
            time.sleep(0.01)
        assert monitor_cleaned.exists()
        assert process.wait(timeout=5) == 143
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
            _process_exists(pid) for pid in tracked_pids
        ):
            time.sleep(0.01)
        assert all(not _process_exists(pid) for pid in tracked_pids)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in tracked_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("initial_signal", "cleanup_signal", "expected_status"),
    [
        (signal.SIGINT, signal.SIGTERM, 130),
        (signal.SIGTERM, signal.SIGINT, 143),
    ],
)
def test_rendered_multi_job_ignores_later_signal_during_stubborn_cleanup(
    tmp_path: Path,
    initial_signal: signal.Signals,
    cleanup_signal: signal.Signals,
    expected_status: int,
):
    (tmp_path / "workdir").mkdir()
    worker = tmp_path / "stubborn-worker.sh"
    worker.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "trap '' INT HUP\n"
        "trap 'touch \"$3\"' TERM\n"
        "(trap '' INT TERM HUP; while true; do sleep 60; done) &\n"
        'echo "$!" > "$2"\n'
        "while true; do wait || true; done\n",
        encoding="utf-8",
    )
    experiment_pid = tmp_path / "experiment.pid"
    experiment_child_pid = tmp_path / "experiment-child.pid"
    cleanup_started = tmp_path / "cleanup-started"
    manifest = ExperimentManifest(
        (
            Experiment(
                "stubborn",
                (
                    "bash",
                    str(worker),
                    str(experiment_pid),
                    str(experiment_child_pid),
                    str(cleanup_started),
                ),
            ),
        )
    )
    script_path = write_rendered_script(
        tmp_path,
        render_multi_job_script(
            valid_multi_config(tmp_path), manifest, "repeated-signal-batch"
        ),
    )
    process = subprocess.Popen(
        ["bash", str(script_path)],
        env={**os.environ, "PBS_JOBID": "12345.pbs1"},
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tracked_pids: list[int] = []

    try:
        deadline = time.monotonic() + 5
        pid_files = (experiment_pid, experiment_child_pid)
        while time.monotonic() < deadline and not all(path.exists() for path in pid_files):
            time.sleep(0.01)
        assert all(path.exists() for path in pid_files)
        tracked_pids = [int(path.read_text(encoding="utf-8")) for path in pid_files]

        process.send_signal(initial_signal)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not cleanup_started.exists():
            time.sleep(0.01)
        assert cleanup_started.exists()
        process.send_signal(cleanup_signal)

        assert process.wait(timeout=5) == expected_status
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
            _process_exists(pid) for pid in tracked_pids
        ):
            time.sleep(0.01)
        assert all(not _process_exists(pid) for pid in tracked_pids)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pid in tracked_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_rendered_multi_job_cleans_descendant_after_leader_was_reaped(
    tmp_path: Path,
):
    (tmp_path / "workdir").mkdir()
    worker = tmp_path / "orphaning-worker.sh"
    worker.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "(trap '' INT TERM HUP; while true; do sleep 60; done) &\n"
        'echo "$!" > "$2"\n',
        encoding="utf-8",
    )
    experiment_pid = tmp_path / "experiment.pid"
    experiment_child_pid = tmp_path / "experiment-child.pid"
    config = valid_multi_config(
        tmp_path,
        setup_commands=signal_after_leader_reaped_setup(),
    )
    manifest = ExperimentManifest(
        (
            Experiment(
                "orphaning",
                (
                    "bash",
                    str(worker),
                    str(experiment_pid),
                    str(experiment_child_pid),
                ),
            ),
        )
    )
    tracked_pids: list[int] = []

    try:
        result = run_rendered_script(
            write_rendered_script(
                tmp_path,
                render_multi_job_script(config, manifest, "orphaning-batch"),
            )
        )
        tracked_pids = [
            int(experiment_pid.read_text(encoding="utf-8")),
            int(experiment_child_pid.read_text(encoding="utf-8")),
        ]

        assert result.returncode == 143
        assert not _process_exists(tracked_pids[0])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _process_exists(tracked_pids[1]):
            time.sleep(0.01)
        assert not _process_exists(tracked_pids[1])
    finally:
        for pid in tracked_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
