# ABCI Generic Job Submitter Design

## Goal

Build a small standalone repository that generates and optionally submits single-node ABCI 3.0 PBS jobs for arbitrary commands and project directories, using the one-GPU `rt_HG` resource by default. The helper must not import, modify, clone, or otherwise depend on the project being executed.

## Scope

The repository supports batch jobs launched from an ABCI login node. It does not connect to ABCI from a workstation, synchronize source code or datasets, provision Python environments, start interactive Open OnDemand sessions, or manage multi-node and multi-GPU distributed jobs.

The default resource type is `rt_HG`, which is ABCI 3.0's shared single-GPU resource. Resource type, walltime, project group, working directory, initialization commands, and monitoring commands remain configurable so the same generator can submit other single-node workloads without code changes. Every generated job requests exactly one node.

## User Interface

The primary command is:

```bash
python submit.py \
  --config configs/abci_default.toml \
  --name example-train \
  -- \
  python -m package.train --epochs 20
```

`submit.py` consumes only its own options before `--`. Every token after `--` is the exact workload command and argument vector. It renders `jobs/<name>.sh`, marks it executable, prints its path, and invokes `qsub` unless `--dry-run` is present. `--dry-run` still performs all validation and rendering.

The accepted helper options are:

- `--config PATH`: required TOML cluster configuration;
- `--name NAME`: required job and generated-file name;
- `--dry-run`: generate without submitting;
- `--print-script`: print the rendered script after writing it; and
- `--`: required separator followed by a non-empty command.

Names accept ASCII letters, digits, dots, underscores, and hyphens, must start with a letter or digit, and have a maximum length of 64 characters. This keeps generated paths and PBS job names predictable.

## Configuration

`configs/abci_example.toml` is committed; `configs/abci_default.toml` is ignored and created by the user.

```toml
group = "your-abci-group"
queue = "rt_HG"
walltime = "12:00:00"
workdir = "/groups/your-abci-group/your-user/your-project"
join_output = true

setup_commands = [
  "source /etc/profile.d/modules.sh",
  "module purge",
  "module load python/3.11",
  "source .venv/bin/activate",
]

[monitor]
enabled = true
interval_seconds = 600
commands = ["nvidia-smi"]
```

Required fields are `group`, `queue`, `walltime`, and `workdir`. `setup_commands` defaults to an empty list. Output joining defaults to true. Monitoring defaults to disabled.

The loader rejects unknown top-level and monitor keys, missing or incorrectly typed fields, non-positive monitoring intervals, unsafe newline-containing values, a relative `workdir`, and a walltime outside `HH:MM:SS`. Setup and monitoring commands are trusted configuration authored by the user; they are emitted as shell statements after newline rejection.

## Generated PBS Script

The committed Jinja template renders:

- a login shell shebang;
- `#PBS -q`, `select`, `walltime`, `-P`, `-N`, and optionally `-j oe`;
- `set -euo pipefail`;
- `cd` to the configured absolute project directory;
- configured setup commands;
- an optional resource-monitoring background loop;
- an EXIT/INT/TERM trap that cleans up background processes; and
- the workload command rendered with POSIX shell quoting for every individual argument.

Configuration values interpolated into PBS directives are strictly validated. Paths and workload arguments are rendered with `shlex.quote`. The command is never assembled from an unquoted raw string.

Generated scripts are runtime artifacts and are ignored except for `jobs/.gitkeep`. A pre-existing `jobs/<name>.sh` is replaced atomically only after configuration and rendering succeed.

## Repository Structure

```text
abci-job/
├── README.md
├── pyproject.toml
├── submit.py
├── abci_job/
│   ├── __init__.py
│   └── submitter.py
├── configs/
│   └── abci_example.toml
├── templates/
│   └── abci.pbs.j2
├── examples/
│   └── commands.md
├── jobs/
│   └── .gitkeep
├── tests/
│   └── test_submitter.py
└── docs/superpowers/specs/
    └── 2026-08-28-abci-generic-job-submitter-design.md
```

The command-line file contains only argument parsing and top-level error reporting. Configuration validation, rendering, atomic writing, executable permissions, and `qsub` invocation live in `abci_job.submitter` as independently testable functions.

## Error Handling and Submission Safety

Validation and rendering happen before any file replacement or scheduler call. Failures use a concise `error: ...` message and a non-zero exit status. A missing `qsub` executable, a rejected submission, or a malformed scheduler response is reported without deleting the generated script, allowing inspection and manual submission.

The submitter prints the generated path before submission and prints `qsub` output, including the returned job identifier. It never invokes `qsub` during `--dry-run`.

## Documentation

The README documents ABCI login-node setup, copying the example config, rendering, submission, inspection with `qstat`, cancellation with `qdel`, and the security model for trusted setup commands.

`examples/commands.md` contains domain-neutral, copyable commands for:

- a Python module invocation;
- a shell script invocation; and
- a command with paths and arguments that require shell quoting.

Every path in those examples is explicit and meant to be edited for the user's ABCI group storage. The examples contain no project-specific package, model, dataset, or method names.

## Testing

Pytest coverage verifies:

- valid TOML loading and defaults;
- rejection of missing, unknown, mistyped, unsafe, and malformed values;
- strict job-name validation;
- POSIX quoting of paths and workload arguments;
- expected `rt_HG` PBS directives and setup commands;
- monitoring loop and cleanup rendering;
- atomic executable script creation;
- `--dry-run` never calling `qsub`; and
- successful and failed `qsub` behavior through an injected command runner.

Tests run locally without PBS, a GPU, ABCI credentials, or an external workload repository.
