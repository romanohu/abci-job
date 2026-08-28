from pathlib import Path

import pytest

from abci_job.submitter import ConfigurationError, load_config, validate_job_name

VALID_CONFIG_TOML = '''
group = "example-group"
queue = "rt_HG"
walltime = "12:00:00"
workdir = "/groups/example-group/user/project"
join_output = true
setup_commands = ["module purge", "source .venv/bin/activate"]

[monitor]
enabled = true
interval_seconds = 600
commands = ["nvidia-smi"]
'''.strip()


def write_config(tmp_path: Path, text: str = VALID_CONFIG_TOML) -> Path:
    path = tmp_path / "abci.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_returns_typed_values(tmp_path: Path):
    config = load_config(write_config(tmp_path))

    assert config.group == "example-group"
    assert config.queue == "rt_HG"
    assert config.walltime == "12:00:00"
    assert config.workdir == Path("/groups/example-group/user/project")
    assert config.join_output is True
    assert config.setup_commands == ("module purge", "source .venv/bin/activate")
    assert config.monitor.enabled is True
    assert config.monitor.interval_seconds == 600
    assert config.monitor.commands == ("nvidia-smi",)


def test_load_config_applies_optional_defaults(tmp_path: Path):
    config = load_config(
        write_config(
            tmp_path,
            '''
group = "example-group"
queue = "rt_HG"
walltime = "12:00:00"
workdir = "/groups/example-group/user/project"
'''.strip(),
        )
    )

    assert config.join_output is True
    assert config.setup_commands == ()
    assert config.monitor.enabled is False
    assert config.monitor.interval_seconds == 0
    assert config.monitor.commands == ()


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
    lines = [line for line in VALID_CONFIG_TOML.splitlines() if not line.startswith(field)]

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
        load_config(write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement)))


@pytest.mark.parametrize("interval", [0, -1])
def test_load_config_rejects_non_positive_monitor_interval(
    tmp_path: Path, interval: int
):
    text = VALID_CONFIG_TOML.replace("interval_seconds = 600", f"interval_seconds = {interval}")

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
        load_config(write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement)))


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
        load_config(write_config(tmp_path, VALID_CONFIG_TOML.replace(original, replacement)))


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
