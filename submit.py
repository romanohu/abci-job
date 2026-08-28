from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from abci_job import (
    ABCIJobError,
    load_config,
    render_job_script,
    submit_job,
    write_job_script,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Generate and submit a single-node ABCI job script."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-script", action="store_true")

    try:
        separator_index = arguments.index("--")
    except ValueError:
        parser.error("a workload command must follow --")

    command = arguments[separator_index + 1 :]
    if not command:
        parser.error("a workload command must follow --")

    args = parser.parse_args(arguments[:separator_index])
    args.command = command
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    submit_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_dir: str | Path = REPOSITORY_ROOT / "jobs",
) -> int:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
        script = render_job_script(config, args.name, args.command)
        job_path = write_job_script(script, args.name, jobs_dir=jobs_dir)
        print(f"Generated job script: {job_path}")
        if args.print_script:
            print(script, end="" if script.endswith("\n") else "\n")
        if args.dry_run:
            return 0
        job_id = submit_job(job_path, runner=submit_runner)
    except (ABCIJobError, OSError) as error:
        error_message = " ".join(str(error).splitlines())
        print(f"error: {error_message}", file=sys.stderr)
        return 2

    print(f"Submitted job: {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
