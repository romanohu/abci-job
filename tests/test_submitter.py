import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from abci_job.submitter import (
    ABCIConfig,
    ConfigurationError,
    MonitorConfig,
    SubmissionError,
    load_config,
    render_job_script,
    resolve_output_path,
    submit_job,
    validate_job_name,
    write_job_script,
)

VALID_CONFIG_TOML = """
group = "example-group"
queue = "rt_HG"
walltime = "12:00:00"
workdir = "/groups/example-group/user/project"
output_path = "logs/custom.log"
join_output = true
setup_commands = ["module purge", "source .venv/bin/activate"]

[monitor]
enabled = true
interval_seconds = 600
commands = ["nvidia-smi"]
""".strip()


def write_config(tmp_path: Path, text: str = VALID_CONFIG_TOML) -> Path:
    path = tmp_path / "abci.toml"
    path.write_text(text, encoding="utf-8")
    return path


def valid_config(**overrides: object) -> ABCIConfig:
    values: dict[str, object] = {
        "group": "example-group",
        "queue": "rt_HG",
        "walltime": "12:00:00",
        "workdir": Path("/groups/example-group/user/project"),
        "output_path": None,
        "join_output": True,
        "setup_commands": (),
        "monitor": MonitorConfig(enabled=False, interval_seconds=0, commands=()),
    }
    values.update(overrides)
    return ABCIConfig(**values)  # type: ignore[arg-type]


def test_render_job_script_quotes_output_path_with_spaces(tmp_path: Path):
    config = valid_config(workdir=Path("/groups/example group/project"))

    script = render_job_script(
        config,
        "example-job",
        ["python", "-m", "package.train", "--output", "results/run one"],
    )

    assert "#PBS -q rt_HG" in script
    assert "#PBS -l select=1" in script
    assert "#PBS -l walltime=12:00:00" in script
    assert "#PBS -P example-group" in script
    assert "#PBS -N example-job" in script
    assert "#PBS -j oe" in script
    assert "cd '/groups/example group/project'" in script
    assert "python -m package.train --output 'results/run one'" in script


def test_render_job_script_quotes_workdir_and_each_command_argument(tmp_path: Path):
    config = valid_config(workdir=Path("/groups/example group/project"))
    script = render_job_script(config, "example-job", ["true"])
    assert "#PBS -o '/groups/example group/project/logs/example-job.log'" in script


@pytest.mark.parametrize("join_output", [True, False])
def test_render_job_script_emits_custom_output_independently_of_joining(
    join_output: bool,
):
    script = render_job_script(
        valid_config(output_path=Path("logs/custom.log"), join_output=join_output),
        "example-job",
        ["true"],
    )
    assert "#PBS -o /groups/example-group/user/project/logs/custom.log" in script
    assert ("#PBS -j oe" in script) is join_output


def test_render_job_script_omits_output_joining_when_disabled():
    script = render_job_script(valid_config(join_output=False), "example-job", ["true"])

    assert "#PBS -j oe" not in script


def test_render_job_script_preserves_setup_command_order():
    script = render_job_script(
        valid_config(setup_commands=("module purge", "source .venv/bin/activate")),
        "example-job",
        ["true"],
    )

    assert script.index("module purge") < script.index("source .venv/bin/activate")


def test_render_job_script_omits_monitor_when_disabled():
    script = render_job_script(valid_config(), "example-job", ["true"])

    assert "while true; do" not in script
    assert "monitor_pid" not in script
    assert "trap cleanup" not in script


def test_render_job_script_renders_monitor_loop_and_cleanup_trap():
    script = render_job_script(
        valid_config(
            monitor=MonitorConfig(
                enabled=True,
                interval_seconds=600,
                commands=("nvidia-smi", "date"),
            )
        ),
        "example-job",
        ["true"],
    )

    assert "while true; do" in script
    assert "nvidia-smi" in script
    assert "date" in script
    assert "sleep 600" in script
    assert "monitor_pid=$!" in script


@pytest.mark.parametrize(
    ("signal_number", "expected_status"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_rendered_monitor_cleans_child_process_and_exits_on_signal(
    tmp_path: Path, signal_number: signal.Signals, expected_status: int
):
    monitor_child_path = tmp_path / "monitor-child.pid"
    monitor_cleanup_path = tmp_path / "monitor-cleaned"
    monitor_command_path = tmp_path / "monitor.sh"
    monitor_command_path.write_text(
        "#!/bin/bash\n"
        'echo "$$" > "$1"\n'
        "trap 'sleep 0.2; touch \"$2\"; exit 0' TERM INT\n"
        "while true; do sleep 1; done\n",
        encoding="utf-8",
    )
    monitor_command = " ".join(
        shlex.quote(argument)
        for argument in (
            "bash",
            str(monitor_command_path),
            str(monitor_child_path),
            str(monitor_cleanup_path),
        )
    )
    script = render_job_script(
        valid_config(
            workdir=tmp_path,
            setup_commands=("workload_wait() { while true; do wait || true; done; }",),
            monitor=MonitorConfig(
                enabled=True,
                interval_seconds=60,
                commands=(monitor_command,),
            ),
        ),
        "signal-test",
        ["workload_wait"],
    )
    script_path = write_job_script(script, "signal-test", jobs_dir=tmp_path)
    process = subprocess.Popen(
        ["bash", str(script_path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    monitor_child_pid: int | None = None
    monitor_process_group: int | None = None

    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not monitor_child_path.exists():
            time.sleep(0.01)
        assert monitor_child_path.exists(), "monitor child did not start"
        monitor_child_pid = int(monitor_child_path.read_text(encoding="utf-8"))
        monitor_process_group = os.getpgid(monitor_child_pid)
        assert monitor_process_group != os.getpgid(process.pid)

        process.send_signal(signal_number)
        assert process.wait(timeout=5) == expected_status
        assert monitor_cleanup_path.exists()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _process_exists(monitor_child_pid):
            time.sleep(0.01)
        assert not _process_exists(monitor_child_pid)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if monitor_process_group is not None:
            try:
                os.killpg(monitor_process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_render_job_script_quotes_metacharacters_quotes_and_empty_tokens():
    script = render_job_script(
        valid_config(),
        "example-job",
        ["printf", "%s", "$HOME; * | &", "it's", ""],
    )

    assert "printf %s '$HOME; * | &' 'it'\"'\"'s' ''" in script


def test_render_job_script_rejects_non_string_command_tokens():
    with pytest.raises(ConfigurationError, match="strings"):
        render_job_script(
            valid_config(),
            "example-job",
            ["printf", 1],  # type: ignore[list-item]
        )


@pytest.mark.parametrize("unsafe_token", ["line\nfeed", "carriage\rreturn"])
def test_render_job_script_rejects_cr_or_lf_in_command_tokens(unsafe_token: str):
    with pytest.raises(ConfigurationError, match="newline"):
        render_job_script(valid_config(), "example-job", ["printf", unsafe_token])


def test_render_job_script_rejects_empty_command():
    with pytest.raises(ConfigurationError, match="command"):
        render_job_script(valid_config(), "example-job", [])


def test_render_job_script_rejects_missing_template(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="template"):
        render_job_script(
            valid_config(),
            "example-job",
            ["true"],
            template_path=tmp_path / "missing.j2",
        )


def test_write_job_script_atomically_replaces_file_and_sets_executable(tmp_path: Path):
    destination = tmp_path / "jobs" / "example.sh"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    result = write_job_script(
        "#!/bin/bash\necho new\n", "example", jobs_dir=destination.parent
    )

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "#!/bin/bash\necho new\n"
    assert destination.stat().st_mode & 0o111 == 0o111
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("failure_point", ["write", "chmod", "replace"])
def test_write_job_script_failure_preserves_existing_file_and_cleans_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
):
    destination = tmp_path / "jobs" / "example.sh"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    if failure_point == "write":
        original_named_temporary_file = tempfile.NamedTemporaryFile

        def failing_named_temporary_file(*args: object, **kwargs: object):
            temporary_file = original_named_temporary_file(*args, **kwargs)

            def fail_write(content: str) -> int:
                raise OSError("write failed")

            temporary_file.write = fail_write
            return temporary_file

        monkeypatch.setattr(
            "abci_job.submitter.tempfile.NamedTemporaryFile",
            failing_named_temporary_file,
        )
    elif failure_point == "chmod":
        monkeypatch.setattr(
            "abci_job.submitter.os.chmod",
            lambda *args: (_ for _ in ()).throw(OSError("chmod failed")),
        )
    else:
        monkeypatch.setattr(
            "abci_job.submitter.os.replace",
            lambda *args: (_ for _ in ()).throw(OSError("replace failed")),
        )

    with pytest.raises(OSError, match=f"{failure_point} failed"):
        write_job_script("new", "example", jobs_dir=destination.parent)

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(destination.parent.iterdir()) == [destination]


def test_write_job_script_keyboard_interrupt_cleans_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "jobs" / "example.sh"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    def interrupt_replace(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("abci_job.submitter.os.replace", interrupt_replace)

    with pytest.raises(KeyboardInterrupt):
        write_job_script("new", "example", jobs_dir=destination.parent)

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(destination.parent.iterdir()) == [destination]


def test_write_job_script_creates_jobs_directory(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"

    result = write_job_script("content", "example", jobs_dir=jobs_dir)

    assert result == jobs_dir / "example.sh"
    assert result.read_text(encoding="utf-8") == "content"


def test_write_job_script_rejects_invalid_name_without_creating_files(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"

    with pytest.raises(ConfigurationError, match="job name"):
        write_job_script("content", "not/a-job", jobs_dir=jobs_dir)

    assert not jobs_dir.exists()


def test_submit_job_uses_expected_scheduler_invocation(tmp_path: Path):
    job_path = tmp_path / "example.sh"
    output_path = tmp_path / "logs" / "example.log"
    calls: list[tuple[object, ...]] = []

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="12345.pbs1\n")

    result = submit_job(job_path, output_path=output_path, runner=runner)

    assert result == "12345.pbs1"
    assert calls == [
        (
            (["qsub", str(job_path)],),
            {"check": True, "capture_output": True, "text": True},
        )
    ]


@pytest.mark.parametrize("stdout", ["", "not a job id\n", "1234 bad\n"])
def test_submit_job_rejects_empty_or_malformed_scheduler_output(
    tmp_path: Path, stdout: str
):
    job_path = tmp_path / "example.sh"
    output_path = tmp_path / "logs" / "example.log"

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, stdout=stdout)

    with pytest.raises(SubmissionError, match="identifier"):
        submit_job(job_path, output_path=output_path, runner=runner)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError(), "qsub executable"),
        (subprocess.CalledProcessError(1, ["qsub"], stderr="denied"), "denied"),
    ],
)
def test_submit_job_wraps_scheduler_errors_and_keeps_script(
    tmp_path: Path, error: Exception, message: str
):
    job_path = tmp_path / "example.sh"
    output_path = tmp_path / "logs" / "example.log"
    job_path.write_text("#!/bin/bash\n", encoding="utf-8")

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise error

    with pytest.raises(SubmissionError, match=message):
        submit_job(job_path, output_path=output_path, runner=runner)

    assert job_path.exists()


def test_submit_job_creates_output_parent_before_scheduler(tmp_path: Path):
    output_path = tmp_path / "remote-workdir" / "logs" / "example.log"

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert output_path.parent.is_dir()
        return subprocess.CompletedProcess(args[0], 0, stdout="12345.pbs1\n")

    result = submit_job(tmp_path / "example.sh", output_path=output_path, runner=runner)

    assert result == "12345.pbs1"


def test_submit_job_reports_output_parent_failure_without_invoking_scheduler(
    tmp_path: Path,
):
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("file", encoding="utf-8")

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("qsub must not run after directory creation fails")

    with pytest.raises(SubmissionError, match="output directory"):
        submit_job(
            tmp_path / "example.sh",
            output_path=blocking_file / "example.log",
            runner=runner,
        )


def test_load_config_returns_typed_values(tmp_path: Path):
    config = load_config(write_config(tmp_path))

    assert config.group == "example-group"
    assert config.queue == "rt_HG"
    assert config.walltime == "12:00:00"
    assert config.workdir == Path("/groups/example-group/user/project")
    assert config.output_path == Path("logs/custom.log")
    assert config.join_output is True
    assert config.setup_commands == ("module purge", "source .venv/bin/activate")
    assert config.monitor.enabled is True
    assert config.monitor.interval_seconds == 600
    assert config.monitor.commands == ("nvidia-smi",)


def test_load_config_applies_optional_defaults(tmp_path: Path):
    config = load_config(
        write_config(
            tmp_path,
            """
group = "example-group"
queue = "rt_HG"
walltime = "12:00:00"
workdir = "/groups/example-group/user/project"
""".strip(),
        )
    )

    assert config.join_output is True
    assert config.setup_commands == ()
    assert config.monitor.enabled is False
    assert config.monitor.interval_seconds == 0
    assert config.monitor.commands == ()
    assert config.output_path is None


def test_resolve_output_path_uses_job_name_default():
    result = resolve_output_path(valid_config(), "example-job")
    assert result == Path("/groups/example-group/user/project/logs/example-job.log")
    assert result.is_absolute()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (
            Path("logs/custom.log"),
            Path("/groups/example-group/user/project/logs/custom.log"),
        ),
        (Path("/shared/logs/custom.log"), Path("/shared/logs/custom.log")),
    ],
)
def test_resolve_output_path_supports_relative_and_absolute_overrides(
    configured: Path, expected: Path
):
    assert resolve_output_path(
        valid_config(output_path=configured), "example-job"
    ) == expected


def test_resolve_output_path_rejects_relative_parent_traversal():
    with pytest.raises(ConfigurationError, match="outside workdir"):
        resolve_output_path(valid_config(output_path=Path("../job.log")), "example-job")


@pytest.mark.parametrize("unsafe", ["bad\x00path", "bad\npath", "bad\rpath"])
def test_resolve_output_path_rejects_control_characters(unsafe: str):
    with pytest.raises(ConfigurationError, match="output_path"):
        resolve_output_path(valid_config(output_path=Path(unsafe)), "example-job")


@pytest.mark.parametrize("name", ["job", "train-01", "eval.v2", "a_b"])
def test_validate_job_name_accepts_scheduler_safe_names(name: str):
    assert validate_job_name(name) == name


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (r'group = "bad\nvalue"', "newline"),
        ('queue = "bad queue"', "queue"),
        ('walltime = "12 hours"', "walltime"),
        ('workdir = "relative/path"', "absolute"),
        ('join_output = "yes"', "join_output"),
        ('setup_commands = "module purge"', "setup_commands"),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path, replacement: str, message: str
):
    keys = {
        "group": 'group = "example-group"',
        "queue": 'queue = "rt_HG"',
        "walltime": 'walltime = "12:00:00"',
        "workdir": 'workdir = "/groups/example-group/user/project"',
        "join_output": "join_output = true",
        "setup_commands": 'setup_commands = ["module purge", "source .venv/bin/activate"]',
    }
    field = replacement.split(" =", maxsplit=1)[0]
    path = write_config(tmp_path, VALID_CONFIG_TOML.replace(keys[field], replacement))

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "name",
    ["", "-leading", "contains space", "contains/slash", "a" * 65, "line\nbreak"],
)
def test_validate_job_name_rejects_unsafe_names(name: str):
    with pytest.raises(ConfigurationError, match="job name"):
        validate_job_name(name)


@pytest.mark.parametrize("field", ["group", "queue", "walltime", "workdir"])
def test_load_config_rejects_missing_required_fields(tmp_path: Path, field: str):
    lines = [
        line for line in VALID_CONFIG_TOML.splitlines() if not line.startswith(field)
    ]

    with pytest.raises(ConfigurationError, match=field):
        load_config(write_config(tmp_path, "\n".join(lines)))


def test_load_config_rejects_unknown_top_level_key(tmp_path: Path):
    text = f'{VALID_CONFIG_TOML}\nunexpected = "value"'

    with pytest.raises(ConfigurationError, match="unexpected"):
        load_config(write_config(tmp_path, text))


def test_load_config_rejects_unknown_monitor_key(tmp_path: Path):
    text = f"{VALID_CONFIG_TOML}\nextra = true"

    with pytest.raises(ConfigurationError, match="monitor"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("group = 1", "group"),
        ("queue = 1", "queue"),
        ("walltime = 1", "walltime"),
        ("workdir = 1", "workdir"),
        ("join_output = 1", "join_output"),
        ("enabled = 1", "monitor.enabled"),
        ("interval_seconds = true", "monitor.interval_seconds"),
    ],
)
def test_load_config_rejects_incorrectly_typed_scalars(
    tmp_path: Path, replacement: str, message: str
):
    field = replacement.split(" =", maxsplit=1)[0]
    original = {
        "group": 'group = "example-group"',
        "queue": 'queue = "rt_HG"',
        "walltime": 'walltime = "12:00:00"',
        "workdir": 'workdir = "/groups/example-group/user/project"',
        "join_output": "join_output = true",
        "enabled": "enabled = true",
        "interval_seconds": "interval_seconds = 600",
    }[field]

    with pytest.raises(ConfigurationError, match=message):
        load_config(
            write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement))
        )


@pytest.mark.parametrize("interval", [0, -1])
def test_load_config_rejects_non_positive_monitor_interval(
    tmp_path: Path, interval: int
):
    text = VALID_CONFIG_TOML.replace(
        "interval_seconds = 600", f"interval_seconds = {interval}"
    )

    with pytest.raises(ConfigurationError, match="monitor.interval_seconds"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('commands = "nvidia-smi"', "monitor.commands"),
        ("commands = [1]", "monitor.commands"),
        (r'commands = ["nvidia-smi\n--loop"]', "newline"),
        (r'setup_commands = ["module purge\nsource .venv/bin/activate"]', "newline"),
    ],
)
def test_load_config_rejects_invalid_command_collections(
    tmp_path: Path, replacement: str, message: str
):
    original = (
        'setup_commands = ["module purge", "source .venv/bin/activate"]'
        if replacement.startswith("setup_commands")
        else 'commands = ["nvidia-smi"]'
    )

    with pytest.raises(ConfigurationError, match=message):
        load_config(
            write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement))
        )


def test_load_config_requires_command_when_monitoring_is_enabled(tmp_path: Path):
    text = VALID_CONFIG_TOML.replace('commands = ["nvidia-smi"]', "commands = []")

    with pytest.raises(ConfigurationError, match="monitor.commands"):
        load_config(write_config(tmp_path, text))


def test_load_config_rejects_zero_walltime(tmp_path: Path):
    text = VALID_CONFIG_TOML.replace('walltime = "12:00:00"', 'walltime = "000:00:00"')

    with pytest.raises(ConfigurationError, match="walltime"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (r'group = "bad\rvalue"', "newline"),
        (r'commands = ["nvidia-smi\r--loop"]', "newline"),
    ],
)
def test_load_config_rejects_carriage_returns(
    tmp_path: Path, replacement: str, message: str
):
    original = (
        'group = "example-group"'
        if replacement.startswith("group")
        else 'commands = ["nvidia-smi"]'
    )

    with pytest.raises(ConfigurationError, match=message):
        load_config(
            write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement))
        )


def test_load_config_rejects_non_table_monitor(tmp_path: Path):
    text = "\n".join(VALID_CONFIG_TOML.splitlines()[:7]) + "\nmonitor = true"

    with pytest.raises(ConfigurationError, match="monitor"):
        load_config(write_config(tmp_path, text))


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("missing.toml", id="missing-file"),
        pytest.param("invalid.toml", id="invalid-toml"),
    ],
)
def test_load_config_wraps_read_errors_with_the_config_path(tmp_path: Path, path: str):
    config_path = tmp_path / path
    if path == "invalid.toml":
        config_path.write_text("group = =", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=path):
        load_config(config_path)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
