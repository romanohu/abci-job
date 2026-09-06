from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from .config import (
    ABCIJobError,
    ConfigurationError,
    EnvironmentConfig,
    JobConfig,
    quote_command,
    validate_job_name,
)

_JOB_ID_PATTERN = re.compile(r"^\d+(?:\.[A-Za-z0-9._-]+)?$")


class SubmissionError(ABCIJobError):
    """Raised when scheduler submission fails."""


def render_job_script(
    environment: EnvironmentConfig,
    job: JobConfig,
    *,
    template_path: str | Path | None = None,
) -> str:
    if not isinstance(environment, EnvironmentConfig) or not isinstance(job, JobConfig):
        raise ConfigurationError("environment and job must be validated configurations")
    path = (
        Path(__file__).resolve().parents[1] / "templates" / "abci.pbs.j2"
        if template_path is None
        else Path(template_path)
    )
    templates = Environment(
        loader=FileSystemLoader(path.parent),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    try:
        template = templates.get_template(path.name)
    except TemplateNotFound as error:
        raise ConfigurationError(f"template {path} was not found") from error

    multiple = len(job.experiments) > 1
    return template.render(
        group=environment.group,
        queue=job.queue,
        rtype=job.rtype,
        walltime=job.walltime,
        job_name=job.name,
        workdir=shlex.quote(str(environment.workdir)),
        log_root=shlex.quote(str(environment.workdir / "logs" / job.name)),
        setup_commands=environment.setup_commands,
        monitor_enabled=environment.monitor.enabled,
        monitor_interval=environment.monitor.interval_seconds,
        monitor_commands=environment.monitor.commands,
        multiple=multiple,
        experiments=tuple(
            {
                "name": experiment.name,
                "gpu": index if multiple else "allocated",
                "command": quote_command(experiment.command),
            }
            for index, experiment in enumerate(job.experiments)
        ),
    )


def write_job_script(content: str, job_name: str, *, jobs_dir: str | Path) -> Path:
    validated_name = validate_job_name(job_name)
    destination_dir = Path(jobs_dir)
    destination = destination_dir / f"{validated_name}.sh"
    temporary_path: Path | None = None

    destination_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination_dir, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o755)
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return destination


def submit_job(
    job_path: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
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
