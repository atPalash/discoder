"""Integration tests for ``ExecutionSandbox.run_command`` (task 5.6).

These exercise the real subprocess machinery (no mocks): a command is launched
through a shell, captured, and either runs to completion or is terminated by the
sandbox. Two behaviours from Requirement 4 are covered:

- **Req 4.3**: the Build_Command runs *within the Job's working directory* --
  the spawned process's current working directory is the sandbox working-copy
  root, so a command that reports its cwd (or touches a relative file) acts on
  that directory.
- **Req 4.8**: a command that runs longer than the configured execution timeout
  is terminated, and the result is flagged as timed out.

Timeouts are kept small (sub-second to a couple of seconds) so the suite stays
fast while still observing real process termination.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from discord_ollama_agent.execution_sandbox import ExecutionSandbox
from discord_ollama_agent.models.job import Job


def _make_job(job_id: str = "demo-target-0001") -> Job:
    """Build a minimal Job for seeding a sandbox working directory."""
    return Job(
        id=job_id,
        target_name="demo-target",
        idea="exercise run_command",
        submitted_at=datetime.now(timezone.utc),
        channel_id=1,
        user_id=1,
    )


def _init_sandbox(tmp_path: Path, source: Path | None = None) -> ExecutionSandbox:
    """Create a sandbox rooted under ``tmp_path`` with an initialized working dir."""
    if source is None:
        source = tmp_path / "source"
        source.mkdir()
    sandbox = ExecutionSandbox(root_dir=tmp_path / "jobs")
    sandbox.init_working_dir(_make_job(), source)
    return sandbox


@pytest.mark.asyncio
async def test_run_command_uses_working_dir_as_cwd(tmp_path: Path):
    """The command runs with the sandbox working dir as its cwd (Req 4.3).

    Runs ``pwd`` and asserts the captured output is the resolved working-copy
    root, proving the spawned process inherited the working dir as its current
    directory rather than the test's cwd.
    """
    sandbox = _init_sandbox(tmp_path)

    result = await sandbox.run_command("pwd", cancel=None, timeout_s=30)

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.cancelled is False
    # The shell's reported cwd resolves to the sandbox working dir.
    reported = Path(result.output.strip()).resolve()
    assert reported == sandbox.working_dir.resolve()


@pytest.mark.asyncio
async def test_run_command_writes_relative_file_in_working_dir(tmp_path: Path):
    """A relative-path command operates inside the working dir (Req 4.3).

    A command that creates a file by relative name lands that file in the
    sandbox working dir, confirming relative paths are interpreted against the
    working-copy root.
    """
    sandbox = _init_sandbox(tmp_path)

    result = await sandbox.run_command(
        "echo hello > artifact.txt", cancel=None, timeout_s=30
    )

    assert result.exit_code == 0
    artifact = sandbox.working_dir / "artifact.txt"
    assert artifact.is_file()
    assert artifact.read_text(encoding="utf-8").strip() == "hello"


@pytest.mark.asyncio
async def test_run_command_terminates_on_timeout(tmp_path: Path):
    """A command exceeding the execution timeout is terminated (Req 4.8).

    Sleeps far longer than the (small) timeout and asserts the sandbox marks the
    result timed out and the process did not exit cleanly with code 0.
    """
    sandbox = _init_sandbox(tmp_path)

    result = await sandbox.run_command("sleep 30", cancel=None, timeout_s=1)

    assert result.timed_out is True
    assert result.cancelled is False
    # A terminated sleep never exits successfully.
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_run_command_terminates_child_process_tree_on_timeout(tmp_path: Path):
    """Timeout termination kills the whole process group, not just the shell.

    The command backgrounds a long sleeper and prints its PID before the shell
    blocks. After the sandbox times out and terminates the group, that child
    must no longer be alive -- otherwise it would outlive the Job (Req 4.8).
    """
    sandbox = _init_sandbox(tmp_path)

    result = await sandbox.run_command(
        "sleep 30 & echo $!; wait", cancel=None, timeout_s=1
    )

    assert result.timed_out is True
    child_pid = int(result.output.strip().splitlines()[0])

    # Give the group-kill a moment to take effect, then confirm the child is gone.
    import asyncio

    await asyncio.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.asyncio
async def test_run_command_completes_before_timeout(tmp_path: Path):
    """A fast command finishes normally without being flagged timed out (Req 4.8).

    Guards against a sandbox that terminates everything: a quick command well
    under the timeout reports its real exit code and is not marked timed out.
    """
    sandbox = _init_sandbox(tmp_path)

    result = await sandbox.run_command("true", cancel=None, timeout_s=10)

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.cancelled is False
