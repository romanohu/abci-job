# PBS output path design

## Goal

Allow each job configuration to select the PBS standard-output destination while
providing a predictable default derived from the job name.

## Configuration

Add an optional top-level TOML field:

```toml
output_path = "logs/custom.log"
```

When omitted, the output path is `logs/<job-name>.log`. Relative paths are
resolved against `workdir`; absolute paths are used unchanged. A relative path
that resolves outside `workdir` is rejected; an explicit absolute path is
required to place output elsewhere. Paths containing a NUL, carriage return, or
newline are rejected.

## Generated PBS script

The renderer emits an absolute output path:

```bash
#PBS -o /groups/example-group/user/project/logs/example-job.log
```

The existing `join_output` behavior remains unchanged. With
`join_output = true`, `#PBS -j oe` merges standard error into this file. With it
disabled, `output_path` applies only to standard output and PBS handles standard
error using its default behavior.

## Submission behavior

Before a real `qsub`, the submitter creates the output file's parent directory.
A dry run only renders and writes the job script; it does not create directories
under `workdir`. A parent-directory creation failure is reported as a concise
submission error and prevents `qsub` from running.

## Documentation and tests

Update the example configuration and README configuration reference. Tests cover
the default path, relative and absolute overrides, unsafe paths, joined and
separate output, parent creation before submission, failure handling, and the
absence of dry-run filesystem side effects.
