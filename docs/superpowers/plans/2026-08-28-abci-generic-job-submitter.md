# ABCI Generic Job Submitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, domain-neutral command-line helper that validates an ABCI configuration, generates an executable single-node PBS script for any command, and optionally submits it with `qsub`.

**Architecture:** Keep scheduler and workload concerns separate. `abci_job.submitter` owns typed configuration, validation, shell-safe rendering, atomic output, and scheduler invocation; the root `submit.py` only parses arguments and reports errors. A committed Jinja template holds the PBS shell structure, while all site- and user-specific values live in an ignored TOML file.

**Tech Stack:** Python 3.11+, standard-library `argparse`, `dataclasses`, `pathlib`, `shlex`, `subprocess`, and `tomllib`; Jinja2 3.1; pytest 8; Ruff.

## Global Constraints

- The public repository must contain no project-specific package, model, dataset, method, or research-domain names.
- The helper runs on an ABCI login node and never initiates SSH or source/data synchronization.
- Every generated job requests exactly one node; `rt_HG` is the example and documented one-GPU default.
- The command after `--` is treated as an argument vector and rendered with POSIX quoting, never as an unquoted raw string.
- Setup and monitoring commands are trusted user-authored shell statements but must not contain newline characters.
- Tests must not require PBS, ABCI credentials, a GPU, or an external workload repository.

---

## File Map

- `pyproject.toml`: package metadata, Jinja dependency, pytest/Ruff development dependencies, and test configuration.
- `.gitignore`: generated scripts, personal configuration, Python environments/caches, PBS output, and macOS artifacts.
- `abci_job/__init__.py`: public exception and function exports.
- `abci_job/submitter.py`: validated configuration types and all generator/submission behavior.
- `templates/abci.pbs.j2`: single-node PBS script structure.
- `submit.py`: user-facing command-line entry point.
- `configs/abci_example.toml`: safe, domain-neutral sample site configuration.
- `jobs/.gitkeep`: retains the otherwise ignored generated-script directory.
- `tests/test_submitter.py`: unit tests for validation, rendering, writing, and scheduler invocation.
- `tests/test_cli.py`: end-to-end CLI contract without calling a real scheduler.
- `README.md`: setup and operational guide.
- `examples/commands.md`: generic command examples only.

---

### Task 1: Typed ABCI configuration and strict validation

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `abci_job/__init__.py`
- Create: `abci_job/submitter.py`
- Create: `configs/abci_example.toml`
- Create: `jobs/.gitkeep`
- Test: `tests/test_submitter.py`

**Interfaces:**
- Produces: `MonitorConfig(enabled: bool, interval_seconds: int, commands: tuple[str, ...])`.
- Produces: `ABCIConfig(group: str, queue: str, walltime: str, workdir: Path, join_output: bool, setup_commands: tuple[str, ...], monitor: MonitorConfig)`.
- Produces: `ABCIJobError`, `ConfigurationError`, and `SubmissionError` exception types.
- Produces: `load_config(path: str | Path) -> ABCIConfig`.
- Produces: `validate_job_name(name: str) -> str`.

- [ ] **Step 1: Create package metadata and the first failing configuration tests**

Create `pyproject.toml` with Python `>=3.11`, runtime dependency `jinja2>=3.1,<4`, development dependencies `pytest>=8,<9` and `ruff>=0.11`, and pytest `pythonpath = ["."]`. Add tests that write TOML under `tmp_path` and specify the desired typed result:

```python
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


def test_load_config_returns_typed_values_and_defaults(tmp_path: Path):
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


@pytest.mark.parametrize("name", ["job", "train-01", "eval.v2", "a_b"])
def test_validate_job_name_accepts_scheduler_safe_names(name):
    assert validate_job_name(name) == name
```

- [ ] **Step 2: Run the focused tests and verify the expected import failure**

Run:

```bash
python -m pytest tests/test_submitter.py -q
```

Expected: collection fails because `abci_job.submitter` does not exist.

- [ ] **Step 3: Implement the minimal typed loader and name validator**

In `abci_job/submitter.py`, define frozen dataclasses, compile `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` for job names, read TOML with `tomllib.load`, apply `join_output=True`, `setup_commands=()`, and a disabled monitor when optional values are absent, and return an `ABCIConfig`. Wrap `OSError` and `tomllib.TOMLDecodeError` as `ConfigurationError` with the config path in the message.

In `abci_job/__init__.py`, export the dataclasses, exception types, loader, and validator through `__all__`.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
python -m pytest tests/test_submitter.py -q
```

Expected: all currently defined tests pass.

- [ ] **Step 5: Add failing validation tests for every rejected input class**

Extend `tests/test_submitter.py` with parameterized cases covering:

```python
@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('group = "bad\\nvalue"', "newline"),
        ('queue = "bad queue"', "queue"),
        ('walltime = "12 hours"', "walltime"),
        ('workdir = "relative/path"', "absolute"),
        ('join_output = "yes"', "join_output"),
        ('setup_commands = "module purge"', "setup_commands"),
    ],
)
def test_load_config_rejects_invalid_values(tmp_path, replacement, message):
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
    ["", "-leading", "contains space", "contains/slash", "a" * 65, "line\\nbreak"],
)
def test_validate_job_name_rejects_unsafe_names(name):
    with pytest.raises(ConfigurationError, match="job name"):
        validate_job_name(name)
```

Also test missing required keys, unknown top-level keys, unknown `[monitor]` keys, boolean values passed where integers are required, non-positive monitor intervals, non-list command collections, non-string commands, newline-containing setup/monitor commands, and `enabled=true` with an empty command list.

- [ ] **Step 6: Run the new tests and verify validation failures**

Run:

```bash
python -m pytest tests/test_submitter.py -q
```

Expected: the new invalid-input tests fail because the initial loader accepts at least one rejected input.

- [ ] **Step 7: Implement strict schema and value validation**

Add private helpers that:

- require exactly `group`, `queue`, `walltime`, and `workdir`, plus optional `join_output`, `setup_commands`, and `monitor`;
- allow only `enabled`, `interval_seconds`, and `commands` inside `[monitor]`;
- reject Python booleans for integer fields;
- require `group` and `queue` to match `^[A-Za-z0-9][A-Za-z0-9._-]*$`;
- require `walltime` to match `^\\d{1,3}:[0-5]\\d:[0-5]\\d$` and contain a non-zero total duration;
- require an absolute `workdir`;
- reject all newline and carriage-return characters in scalar strings and trusted commands; and
- require at least one monitor command when monitoring is enabled.

Use field-specific `ConfigurationError` messages so every test can identify the rejected field.

- [ ] **Step 8: Add the public example configuration and ignore rules**

Create `configs/abci_example.toml` using only neutral placeholder values, `queue = "rt_HG"`, `walltime = "12:00:00"`, two generic environment setup statements, and a disabled monitor with `nvidia-smi` retained as the sample command.

Create `.gitignore` with:

```gitignore
/configs/abci_default.toml
/jobs/*.sh
!/jobs/.gitkeep
*.o[0-9]*
*.e[0-9]*
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
.DS_Store
```

- [ ] **Step 9: Verify Task 1 and commit**

Run:

```bash
python -m pytest tests/test_submitter.py -q
python -m ruff check abci_job tests/test_submitter.py
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and Git reports no whitespace errors.

Commit:

```bash
git add pyproject.toml .gitignore abci_job configs/abci_example.toml jobs/.gitkeep tests/test_submitter.py
git commit -m "feat: validate ABCI job configuration"
```

---

### Task 2: Shell-safe PBS rendering, atomic output, and scheduler submission

**Files:**
- Create: `templates/abci.pbs.j2`
- Modify: `abci_job/submitter.py`
- Modify: `abci_job/__init__.py`
- Test: `tests/test_submitter.py`

**Interfaces:**
- Consumes: `ABCIConfig`, `ConfigurationError`, `SubmissionError`, and `validate_job_name` from Task 1.
- Produces: `render_job_script(config: ABCIConfig, job_name: str, command: Sequence[str], *, template_path: str | Path | None = None) -> str`.
- Produces: `write_job_script(content: str, job_name: str, *, jobs_dir: str | Path) -> Path`.
- Produces: `submit_job(job_path: str | Path, *, runner: Callable[..., CompletedProcess[str]] = subprocess.run) -> str`.

- [ ] **Step 1: Write failing rendering tests**

Add a reusable `valid_config()` fixture and tests asserting that rendered output contains exact PBS directives and quoted command/path values:

```python
def test_render_job_script_quotes_workdir_and_each_command_argument(tmp_path):
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
```

Add separate tests for `join_output=false`, setup-command order, monitor-disabled omission, monitor-enabled loop/interval/commands/cleanup trap, empty command rejection, and a missing template.

- [ ] **Step 2: Run the rendering tests and verify the missing API failure**

Run:

```bash
python -m pytest tests/test_submitter.py -k render -q
```

Expected: import or attribute failures because rendering is not implemented.

- [ ] **Step 3: Create the PBS template and minimal renderer**

Create `templates/abci.pbs.j2` with a `#!/bin/bash -l` shebang, fixed `select=1`, validated directives, `set -euo pipefail`, quoted `cd`, setup statements, optional monitor loop, background PID cleanup trap, and the pre-quoted workload command.

Implement `render_job_script` using a Jinja environment with `StrictUndefined`, `trim_blocks=True`, `lstrip_blocks=True`, and `keep_trailing_newline=True`. Resolve the default template from the repository root, validate the job name again, reject an empty command or non-string/newline-containing command arguments, and pass only validated or pre-quoted values to Jinja.

- [ ] **Step 4: Run rendering tests and verify they pass**

Run:

```bash
python -m pytest tests/test_submitter.py -k render -q
```

Expected: all rendering tests pass.

- [ ] **Step 5: Write failing atomic-output tests**

Add tests that verify:

```python
def test_write_job_script_atomically_replaces_file_and_sets_executable(tmp_path):
    destination = tmp_path / "jobs" / "example.sh"
    destination.parent.mkdir()
    destination.write_text("old", encoding="utf-8")

    result = write_job_script("#!/bin/bash\necho new\n", "example", jobs_dir=destination.parent)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "#!/bin/bash\necho new\n"
    assert destination.stat().st_mode & 0o111 == 0o111
    assert list(destination.parent.iterdir()) == [destination]
```

Also test automatic `jobs_dir` creation and that an invalid name does not create files.

- [ ] **Step 6: Run output tests and verify the missing API failure**

Run:

```bash
python -m pytest tests/test_submitter.py -k write_job -q
```

Expected: import or attribute failures because atomic writing is not implemented.

- [ ] **Step 7: Implement atomic executable output**

Create the destination directory, write UTF-8 through `tempfile.NamedTemporaryFile` in that directory, flush and `os.fsync`, set mode `0o755`, then replace the destination with `os.replace`. On any failure before replacement, unlink the temporary file and leave an existing destination untouched.

- [ ] **Step 8: Write failing scheduler tests**

Use a recording runner function instead of mocking `subprocess.run` globally. Test exact invocation `['qsub', str(job_path)]`, `check=True`, `capture_output=True`, and `text=True`; accept `12345.pbs1`; reject empty or malformed stdout; and convert `FileNotFoundError` and `subprocess.CalledProcessError(stderr='denied')` into `SubmissionError` while keeping the script file.

- [ ] **Step 9: Run scheduler tests and verify the missing API failure**

Run:

```bash
python -m pytest tests/test_submitter.py -k submit_job -q
```

Expected: import or attribute failures because scheduler submission is not implemented.

- [ ] **Step 10: Implement scheduler submission and exported APIs**

Call the injected runner with the exact argument vector and validate stripped stdout against `^\\d+(?:\\.[A-Za-z0-9._-]+)?$`. Include scheduler stderr in `SubmissionError` without a traceback. Export the three new public functions from `abci_job.__init__`.

- [ ] **Step 11: Verify Task 2 and commit**

Run:

```bash
python -m pytest tests/test_submitter.py -q
python -m ruff check abci_job tests/test_submitter.py
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and Git reports no whitespace errors.

Commit:

```bash
git add templates/abci.pbs.j2 abci_job tests/test_submitter.py
git commit -m "feat: render and submit PBS jobs"
```

---

### Task 3: CLI, public documentation, and complete verification

**Files:**
- Create: `submit.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Create: `examples/commands.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `load_config`, `render_job_script`, `write_job_script`, `submit_job`, and `ABCIJobError` from Tasks 1–2.
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`.
- Produces: `main(argv: Sequence[str] | None = None, *, submit_runner=subprocess.run, jobs_dir: str | Path = REPOSITORY_ROOT / "jobs") -> int`.

- [ ] **Step 1: Write failing CLI argument tests**

Create `tests/test_cli.py` and specify parsing behavior:

```python
def test_parse_args_preserves_every_command_token_after_separator():
    args = parse_args(
        [
            "--config", "configs/abci_default.toml",
            "--name", "example-job",
            "--dry-run",
            "--",
            "python", "-m", "package.train", "--label", "run one",
        ]
    )

    assert args.config == Path("configs/abci_default.toml")
    assert args.name == "example-job"
    assert args.dry_run is True
    assert args.command == ["python", "-m", "package.train", "--label", "run one"]
```

Add tests requiring `--config`, `--name`, the separator, and a non-empty command. Verify `--print-script` and reject helper options placed after `--` as ordinary workload arguments only if the command itself is empty.

- [ ] **Step 2: Run CLI tests and verify the missing module failure**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: import failure because root `submit.py` does not exist.

- [ ] **Step 3: Implement argument parsing only**

Use `argparse` for helper options and explicitly split the incoming argv at the first `--`, ensuring the separator and at least one following token exist. Parse only the prefix and store the untouched suffix as `args.command`.

- [ ] **Step 4: Run parsing tests and verify they pass**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: parsing tests pass; end-to-end tests have not yet been added.

- [ ] **Step 5: Write failing end-to-end CLI tests**

Add tests using a real temporary TOML file and jobs directory. The CLI should allow an internal `--jobs-dir` argument only through the callable `main` test seam, not as a public option; implement this as a keyword-only `jobs_dir` defaulting to the repository `jobs/` directory. Verify:

- dry-run writes an executable script and never invokes the injected runner;
- normal mode invokes the runner once and prints both generated path and job ID;
- `--print-script` prints the rendered content;
- invalid configuration and scheduler errors print one `error: ...` line to stderr and return `2`; and
- no traceback is emitted for expected user errors.

- [ ] **Step 6: Run end-to-end CLI tests and verify the orchestration failures**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: new tests fail because `main` does not orchestrate rendering and submission.

- [ ] **Step 7: Implement minimal CLI orchestration**

Load config, render, atomically write, print `Generated job script: <path>`, optionally print the script, return immediately for dry-run, otherwise submit and print `Submitted job: <id>`. Catch `ABCIJobError` and expected filesystem errors, print a concise message to stderr, and return status `2`. Use `raise SystemExit(main())` only in the `if __name__ == '__main__'` block.

- [ ] **Step 8: Add neutral documentation and package script entry point**

Write `README.md` with:

- scope and ABCI 3.0 `rt_HG` single-GPU explanation;
- login-node installation using `python -m venv .venv`, activation, and `python -m pip install -e .`;
- copying `configs/abci_example.toml` to the ignored default file;
- one generic dry-run command and one submission command;
- generated-script inspection and manual `qsub` fallback;
- `qstat` and `qdel` operations;
- configuration field reference;
- trusted-shell-command security note; and
- links to the ABCI 3.0 job execution documentation.

Write `examples/commands.md` with only `python -m package.train`, `bash scripts/run.sh`, and quoting-focused neutral examples. Do not name any real external repository, package, model, dataset, algorithm, or research field.

Keep `python submit.py` as the sole public entry point so the repository does not require installation as a console-script package. The `pyproject.toml` remains responsible only for dependency and test/tool configuration.

- [ ] **Step 9: Run the complete automated verification**

Run:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
```

Expected: all tests pass, Ruff checks and formatting pass, and Git reports no whitespace errors.

- [ ] **Step 10: Perform a real dry-run smoke test without PBS**

Copy the example config to a temporary location, replace `workdir` with the absolute repository path, and run:

```bash
python submit.py \
  --config /tmp/abci-job-smoke.toml \
  --name smoke-test \
  --dry-run \
  --print-script \
  -- \
  python -m package.train --output "results/run one"
```

Expected: exit status `0`, an executable `jobs/smoke-test.sh`, correctly quoted output, `#PBS -q rt_HG`, and no attempt to execute `qsub`.

- [ ] **Step 11: Scan public content for project-specific terminology**

Run:

```bash
rg -ni "model|dataset|algorithm|research|benchmark" README.md examples configs templates abci_job submit.py tests || true
```

Inspect every match and replace domain-specific usage. Generic implementation terms such as an internal software “data model” are allowed only when they cannot identify an external project or research subject.

- [ ] **Step 12: Commit the completed public interface**

```bash
git add README.md pyproject.toml submit.py tests/test_cli.py examples/commands.md
git commit -m "feat: add generic ABCI submission CLI"
```

- [ ] **Step 13: Verify the clean repository state and commit history**

Run:

```bash
git status --short
git log --oneline --decorate -5
python -m pytest -q
```

Expected: empty status output, three implementation commits after the design/plan commits, and a fully passing test suite.
