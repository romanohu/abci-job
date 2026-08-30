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
