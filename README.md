# ABCI Job Submitter

`abci-job` generates a single-node PBS script for either one arbitrary command
or one to eight independent single-GPU commands, and can submit that script
from an ABCI login node. It does not transfer files, open an SSH connection, or
require a particular source tree.

The included example uses ABCI 3.0's `rt_HG` queue for a one-GPU job. Refer to
the [ABCI 3.0 job execution documentation](https://docs.abci.ai/v3/en/job-execution/)
for current queue and resource details.

## Installation

On an ABCI login node, clone this repository and create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Create a local configuration from the committed example:

```bash
cp configs/abci_example.toml configs/abci_default.toml
```

`configs/abci_default.toml` is ignored by Git. Update its group, work directory,
and any environment setup statements for your account.

## Submit a command

Use `--` to separate submitter options from the command to run. A dry run writes
the script but does not call `qsub`:

```bash
python submit.py \
  --config configs/abci_default.toml \
  --name example-job \
  --dry-run \
  -- \
  python -m package.train --output "results/run one"
```

Omit `--dry-run` to submit the generated script:

```bash
python submit.py \
  --config configs/abci_default.toml \
  --name example-job \
  -- \
  bash scripts/run.sh
```

The script is written to `jobs/<name>.sh`. Add `--print-script` to print it as
well. You can inspect and submit a generated script yourself when needed:

Before a manual submission, create the parent directory shown by the rendered
`#PBS -o` path. The normal non-dry-run helper does this automatically.

```bash
sed -n '1,240p' jobs/example-job.sh
qsub jobs/example-job.sh
```

Use the scheduler directly to inspect or cancel work:

```bash
qstat
qdel <job-id>
```

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
written to `logs/<job-name>/<PBS_JOBID>/<experiment-name>.log` under
`workdir`. All entries run to completion; after they finish, any failed entry
makes the PBS job fail. The PBS-level output contains the launcher summary and
optional monitor output.

## Configuration reference

| Field | Required | Description |
| --- | --- | --- |
| `group` | Yes | Scheduler group identifier. |
| `queue` | Yes | Queue name, such as `rt_HG`. |
| `walltime` | Yes | Limit in `HHH:MM:SS` form. |
| `workdir` | Yes | Absolute directory used by the generated script. |
| `output_path` | No | Standard-output file; defaults to `logs/<job-name>.log` under `workdir`. |
| `join_output` | No | Join standard output and standard error; defaults to `true`. |
| `setup_commands` | No | Ordered trusted shell statements run before the command. |
| `monitor.enabled` | No | Enable the monitoring loop; defaults to `false`. |
| `monitor.interval_seconds` | No | Positive interval required when monitoring is enabled. |
| `monitor.commands` | No | Trusted shell statements executed by the monitoring loop. |

Relative `output_path` values are resolved under `workdir`; use an absolute path
to write elsewhere. On a real submission, the helper creates the output file's
parent directory before calling `qsub`. A dry run renders the absolute `#PBS -o`
path but does not create that directory. With `join_output = true`, standard
error is joined into the same file; otherwise PBS applies its default standard-
error handling.

The command after `--` is treated as an argument vector and each token is
POSIX-quoted in the generated script. In contrast, `setup_commands` and
`monitor.commands` are copied as trusted shell statements. Review local
configuration files before submission and do not use untrusted input in those
fields.
