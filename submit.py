from __future__ import annotations

import argparse
import secrets
import string
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from abci_job import (
    ABCIJobError,
    load_environment,
    load_job,
    render_job_script,
    submit_job,
    write_job_script,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an ABCI job script, optionally submitting it with qsub.",
        epilog="Use -- COMMAND [ARG ...] to replace the command in the job settings.",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "environment.toml",
        help="Environment settings (default: configs/environment.toml in this repository).",
    )
    parser.add_argument(
        "--job",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "job.toml",
        help="Job settings (default: configs/job.toml in this repository).",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit the generated script with qsub."
    )
    parser.add_argument(
        "--print-script", action="store_true", help="Print the generated script."
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = None
    if "--" in arguments:
        separator = arguments.index("--")
        command = arguments[separator + 1 :]
        arguments = arguments[:separator]
        if not command:
            parser.error("expected a command after --")
    args = parser.parse_args(arguments)
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
        environment = load_environment(args.environment)
        job = load_job(args.job, command=args.command)
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        random_suffix = "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(8)
        )
        suffix = f"_{timestamp}_{random_suffix}"
        job = replace(job, name=f"{job.name[:64 - len(suffix)]}{suffix}")
        script = render_job_script(environment, job)
        job_path = write_job_script(script, job.name, jobs_dir=jobs_dir)
        print(f"Generated job script: {job_path}")
        if args.print_script:
            print(script, end="" if script.endswith("\n") else "\n")
        if not args.submit:
            return 0
        job_id = submit_job(job_path, runner=submit_runner)
    except (ABCIJobError, OSError) as error:
        error_message = " ".join(str(error).splitlines())
        print(f"error: {error_message}", file=sys.stderr)
        return 2

    print(f"Submitted job: {job_id}")
    log_directory = environment.workdir / "logs" / job.name / job_id
    print(f"Logs (created when the job starts): {log_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
