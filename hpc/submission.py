"""Small, explicit PBS submission primitive used by portable planners."""
from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Sequence


_JOB_ID = re.compile(r"\d+\.[\w.-]+")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _job_id(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    job = lines[-1] if lines else ""
    if not _JOB_ID.fullmatch(job):
        raise RuntimeError(f"unexpected qsub output: {stdout}{stderr}")
    return job


def submit_qsub(
    arguments: Sequence[str], *, host: str = "zeus", timeout: float = 15.0
) -> str:
    """Submit one PBS command once, locally or through an explicit SSH host.

    ``host='local'`` invokes the local ``qsub``.  Other hosts use the fixed
    remote qsub path; the complete remote command is shell-quoted as one SSH
    command argument.  Failures deliberately propagate: retrying could create
    duplicate jobs when the scheduler accepted a request before a disconnect.
    """
    args = [str(argument) for argument in arguments]
    if not _HOST.fullmatch(host) and host != "local":
        raise ValueError(f"invalid submit host: {host!r}")
    if timeout <= 0:
        raise ValueError("submission timeout must be positive")
    if host == "local":
        command = ["qsub", *args]
    else:
        remote_command = shlex.join(["/usr/local/bin/qsub", *args])
        command = [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            host,
            remote_command,
        ]
    result = subprocess.run(
        command, text=True, capture_output=True, check=True, timeout=timeout
    )
    return _job_id(result.stdout, result.stderr)
