"""Example tests for Execution_Sandbox failure paths (task 5.7).

These example/unit tests exercise the two sandbox failure modes that map to
distinct Job failure reasons:

- A failure to initialize the per-Job working directory surfaces
  :class:`SandboxInitError`, which the Coding_Agent records as a
  sandbox-initialization-failure reason (Req 4.10).
- A failure to write a file entry into the working directory surfaces
  :class:`WriteError`, which the Coding_Agent records as a write-failure reason
  (Req 4.11).

Both are forced with real filesystem conditions (no mocking): an init root whose
parent is a regular file (so the directory cannot be created), and a write whose
parent path component is an existing file (so the parent directory cannot be
created).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from discord_ollama_agent.errors import SandboxInitError, WriteError
from discord_ollama_agent.execution_sandbox import ExecutionSandbox
from discord_ollama_agent.models.job import Job


def _make_job(**overrides) -> Job:
    base = dict(
        id="svc-1",
        target_name="svc",
        idea="add a healthcheck endpoint",
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=42,
        user_id=7,
    )
    base.update(overrides)
    return Job(**base)


def test_init_working_dir_failure_raises_sandbox_init_error(tmp_path: Path):
    """A working-dir root that cannot be created surfaces SandboxInitError (Req 4.10).

    The configured root is placed *underneath* an existing regular file, so the
    ``mkdir(parents=True)`` performed by ``init_working_dir`` fails with an OS
    error (the file is not a directory). That failure must be mapped to the
    sandbox-specific :class:`SandboxInitError`.
    """
    blocking_file = tmp_path / "not_a_dir"
    blocking_file.write_text("i am a file", encoding="utf-8")
    # root_dir's parent ("not_a_dir") is a file, so the root cannot be created.
    bad_root = blocking_file / "roots"

    sandbox = ExecutionSandbox(root_dir=bad_root)
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(SandboxInitError):
        sandbox.init_working_dir(_make_job(), source)


def test_init_working_dir_failure_message_identifies_job(tmp_path: Path):
    """The SandboxInitError message references the Job whose init failed (Req 4.10)."""
    blocking_file = tmp_path / "blocker"
    blocking_file.write_text("file", encoding="utf-8")
    sandbox = ExecutionSandbox(root_dir=blocking_file / "roots")

    with pytest.raises(SandboxInitError) as exc_info:
        sandbox.init_working_dir(_make_job(id="svc-init-fail"), tmp_path / "missing")

    assert "svc-init-fail" in str(exc_info.value)


def test_write_file_failure_raises_write_error(tmp_path: Path):
    """A write whose parent path is a file surfaces WriteError (Req 4.11).

    After a normal sandbox init, an in-bounds destination is chosen whose parent
    component is an existing regular file. ``write_file`` resolves the path
    inside the working dir (no escape), then fails creating the parent directory;
    that OS error must be mapped to :class:`WriteError`.
    """
    source = tmp_path / "source"
    source.mkdir()
    sandbox = ExecutionSandbox(root_dir=tmp_path / "roots")
    sandbox.init_working_dir(_make_job(), source)

    # Create a regular file inside the working dir, then try to write "through"
    # it as if it were a directory.
    sandbox.write_file("output", "i am a file, not a directory")
    assert (sandbox.working_dir / "output").is_file()

    with pytest.raises(WriteError):
        sandbox.write_file("output/nested.txt", "should fail")


def test_write_file_failure_message_identifies_path(tmp_path: Path):
    """The WriteError message references the offending relative path (Req 4.11)."""
    source = tmp_path / "source"
    source.mkdir()
    sandbox = ExecutionSandbox(root_dir=tmp_path / "roots")
    sandbox.init_working_dir(_make_job(), source)

    sandbox.write_file("blocker", "file content")

    with pytest.raises(WriteError) as exc_info:
        sandbox.write_file("blocker/child.txt", "should fail")

    assert "blocker/child.txt" in str(exc_info.value)
