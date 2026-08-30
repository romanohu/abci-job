# Multi-GPU Experiment Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated CLI that reserves one ABCI `rt_HF` node and concurrently runs one to eight independent commands, each bound to one GPU with an individual job-ID-scoped log.

**Architecture:** Keep the existing single-command path unchanged. A new `abci_job.experiments` module owns immutable manifest types, strict TOML validation, the `rt_HF` guard, and rendering inputs for a dedicated Bash/Jinja PBS template; `submit_many.py` coordinates those pieces using the existing script writer and scheduler submitter.

**Tech Stack:** Python 3.11+, standard-library `argparse`, `dataclasses`, `pathlib`, `shlex`, `subprocess`, and `tomllib`; Bash; Jinja2 3.1; pytest 8; Ruff.

## Global Constraints

- Reserve exactly one ABCI 3.0 `rt_HF` node with `#PBS -l select=1`; reject every other configured queue before writing a script.
- Accept exactly one to eight experiments and reject zero or more than eight.
- Assign manifest entry `i` to `CUDA_VISIBLE_DEVICES=i`; do not add dynamic GPU reuse, multi-GPU training, CPU affinity, or additional nodes.
- Keep the account-specific ABCI configuration separate from the versionable experiment manifest.
- Each experiment has exactly `name` and `command`; names are unique and scheduler-safe, and commands are non-empty string argument vectors with no CR/LF.
- Execute common `workdir` and `setup_commands` once; do not add per-experiment environment, setup, workdir, output, or resource fields.
- Combine each experiment's stdout/stderr in `<workdir>/logs/<job-name>/<PBS_JOBID>/<experiment-name>.log`.
- Launch every experiment before waiting. A child failure does not stop peers; exit 0 only when all succeed and exit 1 after all finish if any failed.
- INT and TERM terminate every active experiment process group and the optional monitor, returning 130 and 143 respectively; EXIT preserves its initiating status.
- Dry runs validate and write only `jobs/<name>.sh`; they do not invoke `qsub` or create anything under `workdir`.
- Real submission creates the existing PBS-level output parent and invokes `qsub` exactly once.
- Preserve the existing `submit.py` CLI and `templates/abci.pbs.j2` behavior.
- Public examples and documentation must contain no project-, model-, dataset-, method-, account-, or research-specific names.
- Add no runtime dependency and require no PBS installation, ABCI credentials, or GPU in tests.

---

## File Map

- `abci_job/experiments.py`: immutable experiment types, strict manifest loading, full-node config validation, and multi-job rendering.
- `abci_job/submitter.py`: exposes its existing command-vector quoting helper for internal reuse without changing quoting behavior.
- `abci_job/__init__.py`: exports the new manifest types, loader, and renderer.
- `templates/abci_multi.pbs.j2`: full-node PBS directives, process-group lifecycle, GPU assignment, individual logs, aggregate status, and optional monitor.
- `submit_many.py`: multi-experiment argument parsing, dry run, script output, and one scheduler submission.
- `experiments/example.toml`: neutral versionable manifest example.
- `tests/test_experiments.py`: manifest, rendering, concurrent-runtime, aggregate-failure, and signal-cleanup tests.
- `tests/test_submit_many.py`: CLI contract and scheduler-boundary tests.
- `README.md`: distinguishes single-GPU and full-node workflows and documents logs/failure behavior.
- `examples/commands.md`: neutral multi-experiment invocation examples.

---

### Task 1: Strict experiment manifest types and loader

**Files:**
- Create: `abci_job/experiments.py`
- Modify: `abci_job/submitter.py:122-148,258-265`
- Modify: `abci_job/__init__.py:1-27`
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `ConfigurationError` and `validate_job_name(name: str) -> str` from `abci_job.submitter`.
- Produces: `quote_command(command: Sequence[str]) -> str` in `abci_job.submitter`, preserving the previous `_quote_command` behavior.
- Produces: `Experiment(name: str, command: tuple[str, ...])`.
- Produces: `ExperimentManifest(experiments: tuple[Experiment, ...])`.
- Produces: `load_experiment_manifest(path: str | Path) -> ExperimentManifest`.

- [ ] **Step 1: Write failing typed-loader tests**

Create `tests/test_experiments.py` with helpers and the valid one/eight-entry cases:

```python
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
```

- [ ] **Step 2: Run the typed-loader tests and verify the import failure**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -q
```

Expected: collection fails because `abci_job.experiments` does not exist.

- [ ] **Step 3: Implement immutable types and the valid loader path**

Rename `_quote_command` to `quote_command` in `abci_job/submitter.py`, update
`render_job_script` to call the renamed helper, and create
`abci_job/experiments.py`:

```python
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
```

Export `Experiment`, `ExperimentManifest`, and `load_experiment_manifest` from
`abci_job/__init__.py`. Do not export `quote_command` from the package root;
it is shared only between package modules.

- [ ] **Step 4: Run the valid loader tests and verify they pass**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -q
```

Expected: 3 tests pass.

- [ ] **Step 5: Add strict-schema regression tests**

Append the following cases:

```python
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
```

- [ ] **Step 6: Run strict-schema regression tests**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -q
```

Expected: all valid-manifest and strict-schema cases pass.

- [ ] **Step 7: Complete validation by reusing `quote_command`**

Ensure `quote_command` retains these exact checks from the old private helper:

```python
def quote_command(command: Sequence[str]) -> str:
    if isinstance(command, (str, bytes)) or not command:
        raise ConfigurationError("command must contain at least one argument")
    if any(not isinstance(argument, str) for argument in command):
        raise ConfigurationError("command arguments must be strings")
    if any("\n" in argument or "\r" in argument for argument in command):
        raise ConfigurationError("command arguments cannot contain a newline")
    return " ".join(shlex.quote(argument) for argument in command)
```

The dataclass constructors invoke it, so both loaded and manually constructed
experiments receive the same command validation.

- [ ] **Step 8: Verify Task 1 and commit**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py tests/test_submitter.py -q
uv run ruff check --no-cache abci_job tests/test_experiments.py tests/test_submitter.py
git diff --check
```

Expected: all selected tests pass, Ruff reports no errors, and Git reports no
whitespace errors.

Commit:

```bash
git add abci_job/__init__.py abci_job/experiments.py abci_job/submitter.py tests/test_experiments.py
git commit -m "feat: validate multi-GPU experiment manifests"
```

---

### Task 2: Concurrent full-node PBS rendering and runtime lifecycle

**Files:**
- Modify: `abci_job/experiments.py`
- Modify: `abci_job/__init__.py`
- Create: `templates/abci_multi.pbs.j2`
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `ABCIConfig`, `MonitorConfig`, `resolve_output_path`, and internal validated-render/template helpers from `abci_job.submitter`.
- Consumes: `ExperimentManifest` and `quote_command` from Task 1.
- Produces: `render_multi_job_script(config: ABCIConfig, manifest: ExperimentManifest, job_name: str, *, template_path: str | Path | None = None) -> str`.

- [ ] **Step 1: Write failing renderer contract tests**

Extend the module's import section, then add a typed config helper:

```python
import os
import shlex
import signal
import subprocess
import sys
import time

from abci_job import ABCIConfig, ConfigurationError, MonitorConfig
from abci_job.experiments import (
    Experiment,
    ExperimentManifest,
    load_experiment_manifest,
    render_multi_job_script,
)


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
    assert f"experiment_log_dir={tmp_path}/workdir/logs/batch-a/\"$PBS_JOBID\"" in script
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
```

- [ ] **Step 2: Run renderer tests and verify the missing API failure**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -k render_multi -q
```

Expected: collection fails because `render_multi_job_script` does not exist.

- [ ] **Step 3: Implement the renderer and exact template inputs**

Add `import shlex` beside the standard-library imports. Replace the existing
single-line `.submitter` import with this grouped import:

```python
from .submitter import (
    ABCIConfig,
    ConfigurationError,
    _load_template,
    _validate_render_config,
    quote_command,
    resolve_output_path,
    validate_job_name,
)
```

Then append this renderer to `abci_job/experiments.py`:

```python
def render_multi_job_script(
    config: ABCIConfig,
    manifest: ExperimentManifest,
    job_name: str,
    *,
    template_path: str | Path | None = None,
) -> str:
    if not isinstance(manifest, ExperimentManifest):
        raise ConfigurationError("manifest must be an ExperimentManifest")
    group, queue, walltime, workdir, join_output, setup_commands, monitor = (
        _validate_render_config(config)
    )
    if queue != "rt_HF":
        raise ConfigurationError("multi-experiment jobs require queue rt_HF")

    validated_name = validate_job_name(job_name)
    output_path = resolve_output_path(config, validated_name)
    path = (
        Path(__file__).resolve().parents[1] / "templates" / "abci_multi.pbs.j2"
        if template_path is None
        else Path(template_path)
    )
    template = _load_template(path)
    experiments = tuple(
        {
            "name": experiment.name,
            "gpu": index,
            "command": quote_command(experiment.command),
        }
        for index, experiment in enumerate(manifest.experiments)
    )

    return template.render(
        group=group,
        queue=queue,
        walltime=walltime,
        job_name=validated_name,
        join_output=join_output,
        output_path=shlex.quote(str(output_path)),
        workdir=shlex.quote(str(workdir)),
        experiment_log_root=shlex.quote(
            str(workdir / "logs" / validated_name)
        ),
        setup_commands=setup_commands,
        monitor_enabled=monitor.enabled,
        monitor_interval=monitor.interval_seconds,
        monitor_commands=monitor.commands,
        experiments=experiments,
    )
```

Export `render_multi_job_script` from `abci_job/__init__.py`.

- [ ] **Step 4: Create the full concurrent PBS template**

Create `templates/abci_multi.pbs.j2` with this complete lifecycle:

```jinja2
#!/bin/bash -l
#PBS -q {{ queue }}
#PBS -l select=1
#PBS -l walltime={{ walltime }}
#PBS -P {{ group }}
#PBS -N {{ job_name }}
#PBS -o {{ output_path }}
{% if join_output %}#PBS -j oe
{% endif %}
set -euo pipefail

cd {{ workdir }}
{% for setup_command in setup_commands %}
{{ setup_command }}
{% endfor %}
: "${PBS_JOBID:?PBS_JOBID is required}"
experiment_log_dir={{ experiment_log_root }}/"$PBS_JOBID"
mkdir -p -- "$experiment_log_dir"

declare -a experiment_pids=()
declare -a experiment_names=()
declare -a experiment_gpus=()
declare -a experiment_logs=()
declare -a experiment_active=()
monitor_pid=""

cleanup_experiments() {
  local index pid
  for index in "${!experiment_pids[@]}"; do
    if [[ "${experiment_active[$index]}" == "1" ]]; then
      pid=${experiment_pids[$index]}
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for index in "${!experiment_pids[@]}"; do
    if [[ "${experiment_active[$index]}" == "1" ]]; then
      pid=${experiment_pids[$index]}
      wait "$pid" 2>/dev/null || true
      experiment_active[$index]=0
    fi
  done
}

cleanup_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill -TERM -- "-$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}

cleanup_all() {
  local exit_status="$1"
  trap - EXIT INT TERM
  cleanup_experiments
  cleanup_monitor
  exit "$exit_status"
}

exit_on_signal() {
  local exit_status="$1"
  trap - EXIT INT TERM
  cleanup_experiments
  cleanup_monitor
  exit "$exit_status"
}

trap 'cleanup_all "$?"' EXIT
trap 'exit_on_signal 130' INT
trap 'exit_on_signal 143' TERM

set -m
{% if monitor_enabled %}
(
  trap 'exit 0' INT TERM
  while true; do
{% for monitor_command in monitor_commands %}
    {{ monitor_command }}
{% endfor %}
    sleep {{ monitor_interval }}
  done
) &
monitor_pid=$!
{% endif %}
{% for experiment in experiments %}
experiment_log="${experiment_log_dir}/{{ experiment.name }}.log"
printf 'Starting %s on GPU %s; log=%s\n' '{{ experiment.name }}' '{{ experiment.gpu }}' "$experiment_log"
(
  export CUDA_VISIBLE_DEVICES={{ experiment.gpu }}
  exec {{ experiment.command }}
) >"$experiment_log" 2>&1 &
experiment_pids+=("$!")
experiment_names+=('{{ experiment.name }}')
experiment_gpus+=('{{ experiment.gpu }}')
experiment_logs+=("$experiment_log")
experiment_active+=(1)
{% endfor %}
set +m

failures=0
for index in "${!experiment_pids[@]}"; do
  pid=${experiment_pids[$index]}
  if wait "$pid"; then
    exit_status=0
  else
    exit_status=$?
  fi
  experiment_active[$index]=0
  if [[ "$exit_status" -eq 0 ]]; then
    printf 'Succeeded %s on GPU %s; log=%s\n' "${experiment_names[$index]}" "${experiment_gpus[$index]}" "${experiment_logs[$index]}"
  else
    failures=$((failures + 1))
    printf 'Failed %s on GPU %s with exit status %s; log=%s\n' "${experiment_names[$index]}" "${experiment_gpus[$index]}" "$exit_status" "${experiment_logs[$index]}"
  fi
done

if [[ "$failures" -ne 0 ]]; then
  printf '%s of {{ experiments|length }} experiments failed\n' "$failures"
  exit 1
fi
printf 'All {{ experiments|length }} experiments succeeded\n'
```

- [ ] **Step 5: Run renderer tests and verify they pass**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -k render_multi -q
```

Expected: both renderer tests pass.

- [ ] **Step 6: Add concurrent success and aggregate-failure regression tests**

Add a script writer and two integration tests. The first uses a file barrier,
so sequential execution would time out and fail rather than merely appearing
concurrent from rendered `&` characters:

```python
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
    return (sys.executable, "-c", program)


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
```

- [ ] **Step 7: Run concurrent runtime regression tests**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -k "runs_concurrently or aggregate_failure" -q
```

Expected: both tests pass. Keep the file barrier as the deterministic proof
that both commands started before either completed; do not replace it with a
sleep-duration assertion.

- [ ] **Step 8: Add INT/TERM process-group and monitor cleanup regression tests**

Add this worker and parameterized signal test:

```python
def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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

        process.send_signal(signal_number)

        assert process.wait(timeout=5) == expected_status
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
```

- [ ] **Step 9: Run signal cleanup regression tests**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -k cleans_experiment_group -q
```

Expected: both cases return the specified status, write both cleanup markers,
and leave none of the tracked processes alive.

- [ ] **Step 10: Audit coordinated cleanup and run all experiment tests**

Use the trap, active-array, negative-process-group `kill`, and wait logic shown
in Step 4. Do not kill only the immediate child PID; the test records a
grandchild specifically to prevent that regression.

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py -q
```

Expected: all manifest, renderer, concurrency, aggregate-failure, and signal
tests pass with no warnings.

- [ ] **Step 11: Verify Task 2 and commit**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_experiments.py tests/test_submitter.py -q
uv run ruff check --no-cache abci_job tests/test_experiments.py
git diff --check
```

Expected: all selected tests pass, Ruff reports no errors, and Git reports no
whitespace errors.

Commit:

```bash
git add abci_job/__init__.py abci_job/experiments.py templates/abci_multi.pbs.j2 tests/test_experiments.py
git commit -m "feat: run independent experiments across eight GPUs"
```

---

### Task 3: Dedicated multi-experiment submission CLI

**Files:**
- Create: `submit_many.py`
- Create: `tests/test_submit_many.py`

**Interfaces:**
- Consumes: `load_config`, `resolve_output_path`, `submit_job`, and `write_job_script` from `abci_job`.
- Consumes: `load_experiment_manifest` and `render_multi_job_script` from Tasks 1 and 2.
- Produces: `parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace`.
- Produces: `main(argv: Sequence[str] | None = None, *, submit_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run, jobs_dir: str | Path = REPOSITORY_ROOT / "jobs") -> int`.

- [ ] **Step 1: Write failing CLI parsing and help tests**

Create `tests/test_submit_many.py`:

```python
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
```

- [ ] **Step 2: Run parsing tests and verify the missing module failure**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_submit_many.py -k "parse_args or help" -q
```

Expected: collection fails because `submit_many.py` does not exist.

- [ ] **Step 3: Implement the dedicated CLI**

Create `submit_many.py`:

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from abci_job import (
    ABCIJobError,
    load_config,
    load_experiment_manifest,
    render_multi_job_script,
    resolve_output_path,
    submit_job,
    write_job_script,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit an eight-GPU ABCI experiment job."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiments", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-script", action="store_true")
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    submit_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_dir: str | Path = REPOSITORY_ROOT / "jobs",
) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
        manifest = load_experiment_manifest(args.experiments)
        script = render_multi_job_script(config, manifest, args.name)
        job_path = write_job_script(script, args.name, jobs_dir=jobs_dir)
        print(f"Generated job script: {job_path}")
        if args.print_script:
            print(script, end="" if script.endswith("\n") else "\n")
        if args.dry_run:
            return 0
        output_path = resolve_output_path(config, args.name)
        job_id = submit_job(job_path, output_path=output_path, runner=submit_runner)
    except (ABCIJobError, OSError) as error:
        error_message = " ".join(str(error).splitlines())
        print(f"error: {error_message}", file=sys.stderr)
        return 2

    print(f"Submitted job: {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run parsing tests and verify they pass**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_submit_many.py -k "parse_args or help" -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Add dry-run, submission, and error-boundary regression tests**

Append:

```python
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
```

- [ ] **Step 6: Run all CLI regression tests**

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_submit_many.py -q
```

Expected: all CLI parsing and boundary tests pass.

- [ ] **Step 7: Audit the coordination flow and verify Task 3**

Use the exact `main` ordering from Step 3: load config, load manifest, render
(including the `rt_HF` guard), write, optionally print, return on dry run, then
resolve the PBS output and submit once.

Run:

```bash
uv run pytest -p no:cacheprovider tests/test_submit_many.py tests/test_cli.py -q
uv run ruff check --no-cache submit_many.py tests/test_submit_many.py
git diff --check
```

Expected: all new and existing CLI tests pass, Ruff reports no errors, and Git
reports no whitespace errors.

- [ ] **Step 8: Commit Task 3**

```bash
git add submit_many.py tests/test_submit_many.py
git commit -m "feat: submit full-node experiment batches"
```

---

### Task 4: Neutral public examples, documentation, and full verification

**Files:**
- Create: `experiments/example.toml`
- Modify: `README.md:1-98`
- Modify: `examples/commands.md:1-33`
- Test: full repository suite

**Interfaces:**
- Documents: the unchanged `submit.py` one-GPU path and the new `submit_many.py` `rt_HF` path.
- Documents: manifest schema, deterministic GPU mapping, job-ID-scoped logs, all-peers-continue failure behavior, dry runs, and monitoring.
- Preserves: all examples remain domain-neutral and contain no personal account values.

- [ ] **Step 1: Add a neutral versionable manifest example**

Create `experiments/example.toml`:

```toml
[[experiments]]
name = "example-a"
command = ["python", "-c", "print('experiment a')"]

[[experiments]]
name = "example-b"
command = ["python", "-c", "print('experiment b')"]
```

- [ ] **Step 2: Document the full-node workflow in the README**

Change the introduction to state that the helper supports either one arbitrary
command or one to eight independent single-GPU commands. Add a section after
the existing single-command examples containing:

````markdown
## Submit independent commands on eight GPUs

ABCI's `rt_HF` resource reserves one complete eight-GPU node. Set
`queue = "rt_HF"` in the local ABCI configuration, then create a separate
experiment manifest:

```toml
[[experiments]]
name = "example-a"
command = ["python", "-c", "print('experiment a')"]
```

Submit or inspect it with:

```bash
python submit_many.py \
  --config configs/abci_default.toml \
  --experiments experiments/example.toml \
  --name example-batch \
  --dry-run \
  --print-script
```

Remove `--dry-run` to call `qsub`. Manifest entries are assigned in order to
`CUDA_VISIBLE_DEVICES=0` through `7`. Their combined stdout/stderr logs are
written to `logs/<job-name>/<PBS_JOBID>/<experiment-name>.log` under `workdir`.
All entries run to completion; after they finish, any failed entry makes the
PBS job fail. The PBS-level output contains the launcher summary and optional
monitor output.
````

Keep the existing manual `#PBS -o` parent-directory warning and configuration
table intact.

- [ ] **Step 3: Add neutral commands to `examples/commands.md`**

Append:

````markdown
## Full-node experiment manifest

Use a separate manifest for one to eight independent single-GPU commands. The
ABCI configuration used here must set `queue = "rt_HF"`.

```bash
python submit_many.py \
  --config configs/abci_default.toml \
  --experiments experiments/example.toml \
  --name example-batch \
  --dry-run \
  --print-script
```
````

- [ ] **Step 4: Verify the committed example and complete repository**

Run:

```bash
uv run python -c "from abci_job import load_experiment_manifest; print(len(load_experiment_manifest('experiments/example.toml').experiments))"
uv run pytest -p no:cacheprovider -q
uv run ruff check --no-cache .
git diff --check
```

Expected:

```text
2
```

The full test suite passes with no warnings, Ruff reports `All checks passed!`,
and Git reports no whitespace errors.

- [ ] **Step 5: Inspect both generated-script interfaces**

Run:

```bash
uv run pytest -p no:cacheprovider \
  tests/test_cli.py::test_main_print_script_outputs_rendered_content \
  tests/test_submit_many.py::test_main_dry_run_writes_script_without_scheduler_or_workdir_side_effects \
  -q
```

Expected: both tests pass, proving the old command interface and new manifest
interface render independently.

- [ ] **Step 6: Commit Task 4**

```bash
git add README.md examples/commands.md experiments/example.toml
git commit -m "docs: explain full-node experiment batches"
```
