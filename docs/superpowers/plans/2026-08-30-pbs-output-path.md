# PBS Output Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable PBS standard-output destination, with a job-name-based default and safe parent-directory creation immediately before submission.

**Architecture:** Extend `ABCIConfig` with an optional raw output path and centralize job-specific absolute path calculation in `resolve_output_path`. The renderer uses that function for `#PBS -o`, while `submit_job` receives the same resolved path and creates its parent before invoking `qsub`; dry runs stop before that mutation.

**Tech Stack:** Python 3.11+, standard-library `dataclasses`, `pathlib`, `os.path`, and `subprocess`; Jinja2 3.1; TOML; pytest 8; Ruff.

## Global Constraints

- `output_path` is an optional top-level TOML field.
- Omitting `output_path` produces `logs/<job-name>.log` under `workdir`.
- Relative paths are resolved against `workdir`; absolute paths are used unchanged.
- A relative path that resolves outside `workdir` is rejected; callers must use an explicit absolute path to place output elsewhere.
- Paths containing NUL, carriage return, or newline are rejected.
- The generated `#PBS -o` operand is absolute.
- Existing `join_output = true` continues to emit `#PBS -j oe`; when false, only standard output uses `output_path`.
- A real submission creates the output parent directory before `qsub`; a dry run does not create directories under `workdir`.
- Directory-creation errors prevent `qsub` and are reported as concise submission errors.
- Preserve the repository's domain-neutral public vocabulary.
- Do not add dependencies or alter unrelated PBS/resource behavior.

---

## File Map

- `abci_job/submitter.py`: stores and validates `output_path`, resolves the job-specific absolute path, renders it, and creates its parent before scheduler submission.
- `abci_job/__init__.py`: exports `resolve_output_path` with the existing public submitter functions.
- `templates/abci.pbs.j2`: emits the resolved path through `#PBS -o`.
- `submit.py`: resolves the path once for the real-submission call while retaining the current dry-run boundary.
- `tests/test_submitter.py`: covers config loading, path resolution, rendering, unsafe paths, and submission-directory failure behavior.
- `tests/test_cli.py`: proves dry runs have no `workdir` side effects and real submission passes the resolved output path.
- `configs/abci_example.toml`: shows the optional setting without forcing one output file for every job name.
- `README.md`: documents the default, relative/absolute semantics, `join_output`, and real-submit directory creation.

---

### Task 1: Output configuration, resolution, and PBS rendering

**Files:**
- Modify: `abci_job/submitter.py:19-187`
- Modify: `abci_job/__init__.py:1-27`
- Modify: `templates/abci.pbs.j2:1-9`
- Test: `tests/test_submitter.py:15-460`

**Interfaces:**
- Produces: `ABCIConfig.output_path: Path | None`.
- Produces: `resolve_output_path(config: ABCIConfig, job_name: str) -> Path` returning an absolute, validated path without creating it.
- Changes: `render_job_script(...) -> str` emits `#PBS -o <absolute-path>` using the same resolver.

- [ ] **Step 1: Write failing config and path-resolution tests**

Add `output_path = "logs/custom.log"` immediately after `workdir` in
`VALID_CONFIG_TOML`, add `"output_path": None` to `valid_config()`, import
`resolve_output_path`, and extend the typed/default assertions:

```python
def test_load_config_returns_typed_values(tmp_path: Path):
    config = load_config(write_config(tmp_path))
    assert config.output_path == Path("logs/custom.log")


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
    assert config.output_path is None
```

Add focused resolver cases:

```python
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
```

- [ ] **Step 2: Run the resolver tests and verify they fail**

Run:

```bash
python -m pytest tests/test_submitter.py -k "output_path or optional_defaults or typed_values" -q
```

Expected: collection fails because `resolve_output_path` is not exported and `ABCIConfig` does not accept `output_path`.

- [ ] **Step 3: Implement the minimal typed field and resolver**

In `abci_job/submitter.py`, admit the optional key and add the dataclass field:

```python
_OPTIONAL_CONFIG_KEYS = {"output_path", "join_output", "setup_commands", "monitor"}


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
```

Load the optional field only after `workdir` has been validated:

```python
configured_output = data.get("output_path")
output_path = (
    None
    if configured_output is None
    else Path(_validate_output_path_string(configured_output))
)

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
```

Add the public resolver and its narrow validation helpers. Use `os.path.abspath` rather than `Path.resolve()` so lexical normalization does not depend on whether ABCI paths exist on the machine performing a dry run:

```python
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


def _validate_output_path_string(value: object) -> str:
    output_path = _validate_string(value, "output_path")
    if "\x00" in output_path:
        raise ConfigurationError("output_path cannot contain NUL")
    return output_path
```

Call `resolve_output_path` from `render_job_script` immediately after
`_validate_render_config` so manually constructed `ABCIConfig` values receive
the same checks. Add `resolve_output_path` to both the import tuple and
`__all__` in `abci_job/__init__.py`:

```python
from .submitter import (
    ABCIConfig,
    ABCIJobError,
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

__all__ = [
    "ABCIConfig",
    "ABCIJobError",
    "ConfigurationError",
    "MonitorConfig",
    "SubmissionError",
    "load_config",
    "render_job_script",
    "resolve_output_path",
    "submit_job",
    "validate_job_name",
    "write_job_script",
]
```

- [ ] **Step 4: Run the resolver tests and verify they pass**

Run:

```bash
python -m pytest tests/test_submitter.py -k "output_path or optional_defaults or typed_values" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing PBS rendering tests**

Extend the main rendering assertion and cover custom paths with spaces plus both `join_output` states:

```python
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
```

- [ ] **Step 6: Run the rendering tests and verify the missing directive failure**

Run:

```bash
python -m pytest tests/test_submitter.py -k render_job_script -q
```

Expected: the new assertions fail because the PBS template has no `#PBS -o` directive.

- [ ] **Step 7: Render the validated absolute output path**

Replace the body of `render_job_script` with the same validated inputs plus a
shell-quoted PBS operand:

```python
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
    template = _load_template(template_path)
    validated_job_name = validate_job_name(job_name)
    output_path = resolve_output_path(config, validated_job_name)

    return template.render(
        group=group,
        queue=queue,
        walltime=walltime,
        job_name=validated_job_name,
        output_path=shlex.quote(str(output_path)),
        join_output=join_output,
        workdir=shlex.quote(str(workdir)),
        setup_commands=setup_commands,
        monitor_enabled=monitor.enabled,
        monitor_interval=monitor.interval_seconds,
        monitor_commands=monitor.commands,
        command=_quote_command(command),
    )
```

Add the directive next to the existing name/join directives in `templates/abci.pbs.j2`:

```jinja2
#PBS -N {{ job_name }}
#PBS -o {{ output_path }}
{% if join_output %}#PBS -j oe
{% endif %}
```

- [ ] **Step 8: Verify Task 1 and commit**

Run:

```bash
python -m pytest tests/test_submitter.py -q
python -m ruff check abci_job tests/test_submitter.py
git diff --check
```

Expected: all submitter tests pass, Ruff reports no errors, and Git reports no whitespace errors.

Commit:

```bash
git add abci_job/__init__.py abci_job/submitter.py templates/abci.pbs.j2 tests/test_submitter.py
git commit -m "feat: configure PBS output paths"
```

---

### Task 2: Submission-time directory creation and dry-run boundary

**Files:**
- Modify: `abci_job/submitter.py:142-163`
- Modify: `submit.py:9-66`
- Test: `tests/test_submitter.py:300-370`
- Test: `tests/test_cli.py:80-280`

**Interfaces:**
- Consumes: `resolve_output_path(config, job_name) -> Path` from Task 1.
- Changes: `submit_job(job_path: str | Path, *, output_path: str | Path, runner: Callable[..., CompletedProcess[str]] = subprocess.run) -> str`.
- Preserves: `main(..., --dry-run)` writes the generated job script and returns before `submit_job` or output-directory creation.

- [ ] **Step 1: Write failing submission-order and failure tests**

Update existing direct `submit_job` calls to pass `output_path`. Add tests proving parent creation precedes the scheduler and a creation failure suppresses it:

```python
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
```

- [ ] **Step 2: Run the direct submission tests and verify signature failures**

Run:

```bash
python -m pytest tests/test_submitter.py -k submit_job -q
```

Expected: the new tests fail because `submit_job` does not accept `output_path` and does not create directories.

- [ ] **Step 3: Create the output parent before invoking `qsub`**

Replace `submit_job` with the following function, keeping scheduler errors and
job identifiers on their current contract:

```python
def submit_job(
    job_path: str | Path,
    *,
    output_path: str | Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    output_directory = Path(output_path).parent
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SubmissionError(
            f"could not create output directory {output_directory}: {error}"
        ) from error

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
```

- [ ] **Step 4: Run direct submission tests and verify they pass**

Run:

```bash
python -m pytest tests/test_submitter.py -k submit_job -q
```

Expected: all selected tests pass and the runner observes the directory before invocation.

- [ ] **Step 5: Write failing CLI tests for the resolved path and dry-run side effects**

First change the shared CLI test helper so every valid config points inside
`tmp_path`, preventing scheduler-error tests from creating `/tmp/logs`:

```python
def write_config(tmp_path: Path) -> Path:
    workdir = tmp_path / "workdir"
    config_path = tmp_path / "abci.toml"
    config_path.write_text(
        CONFIG_TOML.replace('workdir = "/tmp"', f'workdir = "{workdir}"'),
        encoding="utf-8",
    )
    return config_path
```

In `test_main_submits_once_and_reports_generated_path_and_job_id`, assert from
inside `runner` that `(tmp_path / "workdir" / "logs").is_dir()`. Add an
isolated dry-run case:

```python
def test_main_dry_run_does_not_create_output_directory_under_workdir(
    tmp_path: Path,
):
    workdir = tmp_path / "workdir"

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
        jobs_dir=tmp_path / "jobs",
    )

    assert result == 0
    assert not workdir.exists()
```

The successful runner assertion is:

```python
def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    assert (tmp_path / "workdir" / "logs").is_dir()
    calls.append((args, kwargs))
    return subprocess.CompletedProcess(args[0], 0, stdout="12345.pbs1\n")
```

- [ ] **Step 6: Run CLI tests and verify the real-submit call fails**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: real-submission tests fail because `main` does not pass `output_path` to the now-required `submit_job` argument; the dry-run side-effect assertion already passes.

- [ ] **Step 7: Pass the resolved output path only after the dry-run return**

Import `resolve_output_path` in `submit.py`. Preserve the existing early return and change only the real-submission branch:

```python
if args.dry_run:
    return 0
output_path = resolve_output_path(config, args.name)
job_id = submit_job(job_path, output_path=output_path, runner=submit_runner)
```

This keeps rendering/path validation active during dry runs but defers `mkdir` until a real submission.

- [ ] **Step 8: Verify Task 2 and commit**

Run:

```bash
python -m pytest tests/test_submitter.py tests/test_cli.py -q
python -m ruff check abci_job submit.py tests
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and Git reports no whitespace errors.

Commit:

```bash
git add abci_job/submitter.py submit.py tests/test_submitter.py tests/test_cli.py
git commit -m "feat: prepare PBS output directory on submit"
```

---

### Task 3: Public configuration documentation and full regression verification

**Files:**
- Modify: `configs/abci_example.toml:1-8`
- Modify: `README.md:29-82`
- Test: full repository suite

**Interfaces:**
- Documents: optional `output_path`, default `logs/<job-name>.log`, relative/absolute resolution, `join_output`, and real-submit directory creation.
- Preserves: the example remains valid without account-specific or research-specific values.

- [ ] **Step 1: Add the optional setting to the example without overriding the default**

Place this comment immediately after `workdir` in `configs/abci_example.toml`:

```toml
# Relative paths are resolved below workdir. Default: logs/<job-name>.log
# output_path = "logs/custom.log"
```

- [ ] **Step 2: Document output behavior in the README**

Add the configuration row:

```markdown
| `output_path` | No | Standard-output file; defaults to `logs/<job-name>.log` under `workdir`. |
```

After the table, add:

```markdown
Relative `output_path` values are resolved under `workdir`; use an absolute path
to write elsewhere. On a real submission, the helper creates the output file's
parent directory before calling `qsub`. A dry run renders the absolute `#PBS -o`
path but does not create that directory. With `join_output = true`, standard
error is joined into the same file; otherwise PBS applies its default standard-
error handling.
```

- [ ] **Step 3: Run the full verification suite**

Run:

```bash
python -m pytest -q
python -m ruff check .
git diff --check
```

Expected: every test passes, Ruff reports no errors, and Git reports no whitespace errors.

- [ ] **Step 4: Inspect the generated script contract manually**

Run:

```bash
python submit.py \
  --config configs/abci_example.toml \
  --name example-job \
  --dry-run \
  --print-script \
  -- \
  true
```

Expected output includes:

```bash
#PBS -N example-job
#PBS -o /groups/your-abci-group/your-user/your-project/logs/example-job.log
#PBS -j oe
```

It must not create `/groups/your-abci-group/your-user/your-project/logs`.

- [ ] **Step 5: Commit the documentation**

```bash
git add README.md configs/abci_example.toml
git commit -m "docs: explain PBS output files"
```
