# Multi-GPU experiment launcher design

## Goal

Add a second submission path that reserves one ABCI 3.0 full GPU node and
runs between one and eight independent, single-GPU commands concurrently.
Each command receives one fixed GPU, writes an individual log, and runs to
completion even if another command fails.

The existing single-command `submit.py` workflow remains unchanged.

## ABCI resource contract

The multi-experiment path requires the ABCI 3.0 resource type `rt_HF` with
`select=1`. This is a node-exclusive allocation with eight GPUs. The existing
`rt_HG` example remains the one-GPU, node-sharing path.

`submit_many.py` rejects a site configuration whose `queue` is not exactly
`rt_HF`. The generated PBS script always emits `#PBS -l select=1`; the user
cannot request additional nodes through the experiment manifest.

Reference: <https://docs.abci.ai/v3/en/job-execution/>

## Command-line interface

Add a dedicated entry point:

```bash
python submit_many.py \
  --config configs/abci_default.toml \
  --experiments experiments/runs.toml \
  --name batch-a
```

Supported options are:

- `--config PATH`: required existing ABCI site configuration;
- `--experiments PATH`: required experiment manifest;
- `--name NAME`: required scheduler-safe PBS job name;
- `--dry-run`: validate and write the script without submitting it; and
- `--print-script`: print the generated script.

Unlike `submit.py`, this command has no `--` workload boundary because every
workload command comes from the manifest.

The CLI reuses the existing `load_config`, `write_job_script`,
`resolve_output_path`, and `submit_job` functions. It writes the generated
script to `jobs/<name>.sh`. A real submission prepares the configured PBS
output parent and invokes `qsub` once. A dry run neither invokes `qsub` nor
creates anything below `workdir`.

All configuration, manifest, rendering, filesystem, and scheduler errors use
the existing single-line `error: ...` format and return status 2 without a
traceback.

## Experiment manifest

The experiment manifest is separate from the account-specific ABCI
configuration so it can be version controlled independently.

```toml
[[experiments]]
name = "run-a"
command = ["python", "-m", "package.train", "seed=1"]

[[experiments]]
name = "run-b"
command = ["python", "-m", "package.train", "seed=2"]
```

The manifest schema is strict:

- the only top-level key is `experiments`;
- `experiments` contains between one and eight tables;
- each table contains exactly `name` and `command`;
- `name` follows the existing scheduler-safe name rule and is unique within
  the manifest;
- `command` is a non-empty array of strings; and
- command arguments containing carriage returns or newlines are rejected.

The command is an argument vector, not trusted shell source. Each token is
POSIX-quoted using the same rules as the existing single-command renderer.
The manifest cannot define per-experiment setup commands, environment
variables, working directories, output paths, resource requests, or a separate
raw-shell field. A caller that intentionally needs shell evaluation can still
make `bash`, `-lc`, and the shell program explicit command-array elements.

Add immutable `Experiment` and `ExperimentManifest` data types plus
`load_experiment_manifest(path) -> ExperimentManifest` in
`abci_job/experiments.py`.

## Generated PBS job

Add `templates/abci_multi.pbs.j2` and a renderer owned by
`abci_job.experiments`. The template emits the same group, walltime, job name,
PBS output, output joining, common work directory, setup commands, and optional
monitoring behavior as the existing template. Its resource directives are
`rt_HF` and `select=1` through the validated site configuration.

At runtime, the script:

1. changes to the common `workdir`;
2. executes the common `setup_commands` once in the parent shell;
3. requires PBS to provide a non-empty `PBS_JOBID`;
4. creates `logs/<job-name>/<PBS_JOBID>/` below `workdir`;
5. starts every experiment as an independent background process group;
6. records each process ID, experiment name, assigned GPU, and log path;
7. waits for every process and records its exit status; and
8. exits 0 only when every experiment succeeded, otherwise exits 1.

Experiment index `i` is assigned `CUDA_VISIBLE_DEVICES=i`, so manifest entries
map in order to physical GPU identifiers 0 through 7. The variable is set only
for that experiment's process group. No CPU affinity, CPU-thread allocation,
multi-GPU training, or dynamic scheduling is added.

Each experiment combines standard output and standard error in:

```text
<workdir>/logs/<job-name>/<PBS_JOBID>/<experiment-name>.log
```

The PBS-level `#PBS -o` remains controlled by the existing `output_path`
setting. It receives launcher summaries: experiment-to-GPU mappings, individual
log paths, completion statuses, and the final success/failure count. Experiment
program output does not interleave in the PBS-level log.

The runtime log directory is created on the compute node because `PBS_JOBID`
does not exist at submission time. Reusing the same `--name` therefore never
overwrites an earlier experiment log directory.

## Failure and signal behavior

A failed experiment does not terminate or cancel its peers. All experiments
are launched before the script begins waiting, and every recorded process is
waited exactly once. After all experiments finish, any non-zero child status
makes the PBS job exit 1.

The generated script installs coordinated INT, TERM, and EXIT handling for the
experiment process groups and the optional monitor. When PBS cancels or
terminates the job, the handler sends TERM to every still-running experiment
process group, terminates the monitor process group, waits for cleanup, and
returns 130 for INT or 143 for TERM. The EXIT handler preserves the status that
caused it to run. Cleanup targets only children still recorded as active,
tolerates children that have already exited, and must not leave descendant
processes running.

Failures before experiment launch, including common setup, missing `PBS_JOBID`,
or log-directory creation, terminate the PBS job immediately with a non-zero
status.

## Files and responsibilities

- `submit_many.py`: parses multi-experiment CLI arguments and coordinates
  validation, rendering, script writing, dry runs, and submission.
- `abci_job/experiments.py`: owns manifest types, strict validation, safe
  command rendering, the `rt_HF` guard, and multi-job rendering inputs.
- `templates/abci_multi.pbs.j2`: owns the PBS shell lifecycle for concurrent
  experiments.
- `experiments/example.toml`: provides only neutral example commands and
  placeholder names.
- `tests/test_experiments.py`: tests schema validation, rendering, GPU/log
  mappings, concurrent execution, aggregate failure, and signal cleanup.
- `tests/test_submit_many.py`: tests CLI parsing, dry-run isolation, scheduler
  invocation, and single-line errors.
- `README.md` and `examples/commands.md`: document the one-GPU and eight-GPU
  workflows without project-, model-, dataset-, or research-specific terms.

Existing files may receive only the exports needed for the new public types and
functions. The single-command CLI and its template retain their current
behavior.

## Testing

Automated tests require no ABCI credentials, PBS installation, or GPU.

Coverage includes:

- valid manifests with one and eight experiments;
- zero, nine, duplicate-name, unknown-key, wrong-type, empty-command, and
  newline-containing-command failures;
- rejection of a non-`rt_HF` site configuration before a script is written;
- token-by-token shell quoting;
- deterministic index-to-GPU and experiment-to-log mapping;
- a generated-script integration test proving multiple commands overlap in
  execution without relying only on rendered text;
- successful aggregate exit when all children succeed;
- non-zero aggregate exit after a failing child while successful peers still
  complete;
- INT and TERM cleanup of child process groups and the optional monitor;
- dry-run script generation without scheduler or `workdir` side effects;
- exactly one `qsub` call for a real submission;
- concise, single-line user-facing errors; and
- the full existing single-command regression suite.

The generated-script tests supply a temporary `workdir` and synthetic
`PBS_JOBID`, use CPU-only commands, and verify the contents and locations of
individual logs.

## Out of scope

- more than eight experiments or queuing waves of experiments;
- automatic reuse of a GPU after an experiment finishes;
- distributed or multi-GPU experiments;
- per-experiment environment, setup, working directory, or resource settings;
- CPU-core pinning or memory partitioning;
- multiple compute nodes;
- changes to the existing one-GPU submission interface; and
- file synchronization, SSH orchestration, or workload-repository integration.
