from __future__ import annotations

import shlex
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

import pytest

HPC = Path(__file__).resolve().parents[1] / "hpc"
sys.path.insert(0, str(HPC))
import submission  # noqa: E402
from submit import cell_id, expanded  # noqa: E402


def completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")


def test_local_submission_uses_local_qsub_and_validates_job_id():
    with patch.object(submission.subprocess, "run", return_value=completed("41.zeus\n")) as run:
        assert submission.submit_qsub(["-N", "local job", "run.pbs"], host="local") == "41.zeus"
    assert run.call_args.args[0] == ["qsub", "-N", "local job", "run.pbs"]
    assert run.call_args.kwargs == {
        "text": True,
        "capture_output": True,
        "check": True,
        "timeout": 15.0,
    }


def test_remote_submission_is_quoted_as_one_ssh_command():
    arguments = ["-v", "KEY=a value;still-one-value", "/tmp/run file.pbs"]
    with patch.object(submission.subprocess, "run", return_value=completed("52.zeus-master\n")) as run:
        assert submission.submit_qsub(arguments, host="zeus", timeout=9) == "52.zeus-master"
    assert run.call_args.args[0] == [
        "ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
        "zeus", shlex.join(["/usr/local/bin/qsub", *arguments]),
    ]
    assert run.call_args.kwargs["timeout"] == 9


@pytest.mark.parametrize("stdout", ["", "qsub: denied\n", "12 not-a-job\n"])
def test_bad_scheduler_output_is_rejected(stdout: str):
    with patch.object(submission.subprocess, "run", return_value=completed(stdout)):
        with pytest.raises(RuntimeError, match="unexpected qsub output"):
            submission.submit_qsub(["run.pbs"], host="local")


def test_timeout_propagates_without_retry():
    with patch.object(
        submission.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(["qsub"], 15),
    ) as run:
        with pytest.raises(subprocess.TimeoutExpired):
            submission.submit_qsub(["run.pbs"], host="local")
    run.assert_called_once()


def test_invalid_remote_host_is_rejected_before_subprocess():
    with pytest.raises(ValueError, match="invalid submit host"):
        submission.submit_qsub(["run.pbs"], host="-oProxyCommand=bad")


def test_existing_planner_helpers_remain_importable():
    assert cell_id(["--seed", "123"]) == cell_id(["--seed", "123"])
    assert expanded("compact_lj_reconstruction", smoke=True)
