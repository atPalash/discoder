"""Integration tests for GitManager.commit_and_push (task 7.7).

Exercises the real :class:`GitManager.commit_and_push` end to end against
temp-directory local Git repositories, with no mocking of GitPython or the git
subprocess:

- Success path (Req 5.3, 5.4, 5.8): a working copy with changes, a target whose
  ``credentials_ref`` resolves from an environment variable, and an ``origin``
  pointed at a local bare repository. The call must create the Job's branch,
  commit, and push it to the remote, returning :attr:`PushStatus.PUSHED` with
  the derived branch name; the branch must actually exist on the remote with the
  committed content. A local ``file://`` / bare remote needs no real
  authentication, so a resolvable (but otherwise unused) credentials reference
  is enough to clear the missing-credentials guard and let the push proceed.

- Failure path (Req 5.5): when the configured remote cannot be reached (a bad
  remote URL), the push fails and the result maps to
  :data:`PUSH_ERROR_REASON` with :attr:`PushStatus.FAILED` and no branch name.

These are example-based integration tests (not a numbered property); they
complement the property tests for the no-changes (7.3) and missing-credentials
(7.4) paths.

Covers Requirements 5.3, 5.4, 5.5, 5.8.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from git import Repo

from discord_ollama_agent.git_manager import (
    PUSH_ERROR_REASON,
    GitManager,
    PushStatus,
)
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget


def _init_work_repo(work_dir: Path) -> Repo:
    """Create a local repo at ``work_dir`` with a committed baseline."""
    work_dir.mkdir(parents=True, exist_ok=True)
    repo = Repo.init(work_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test Agent")
        cw.set_value("user", "email", "agent@example.invalid")
    (work_dir / "README.md").write_text("seed\n", encoding="utf-8")
    repo.git.add("--all")
    repo.index.commit("initial commit")
    return repo


def _make_job(job_id: str, target_name: str, status: JobStatus = JobStatus.RUNNING) -> Job:
    return Job(
        id=job_id,
        target_name=target_name,
        idea="add a feature file",
        status=status,
        submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        channel_id=42,
        user_id=7,
    )


async def test_commit_and_push_new_branch_succeeds(monkeypatch):
    """A dirty working copy is committed and pushed; status is PUSHED (5.3, 5.4, 5.8).

    The target's ``credentials_ref`` names an environment variable that is set,
    so credential resolution yields a value and the missing-credentials guard is
    cleared. The local bare remote needs no real authentication, so the push
    proceeds and the Job's derived branch lands on the remote carrying the new
    commit and file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work_dir = root / "work"
        remote_dir = root / "remote.git"

        # A real local bare repository serves as the push target ("file" remote).
        Repo.init(remote_dir, bare=True)

        repo = _init_work_repo(work_dir)
        working_copy = Path(repo.working_tree_dir)

        # Make the working tree dirty: there is something to commit and push.
        (working_copy / "feature.txt").write_text("a generated change\n", encoding="utf-8")

        # A resolvable credentials reference: an env var that is set. The value
        # is never used by a local remote but it clears the credentials guard.
        creds_env = f"GIT_TOKEN_{uuid.uuid4().hex.upper()}"
        monkeypatch.setenv(creds_env, "unused-but-present-token")

        target = RegisteredTarget(
            name="svc",
            directory_path=working_copy,
            repo_remote=str(remote_dir),
            default_branch="main",
            branch_scheme="agent/{job_id}",
            credentials_ref=creds_env,
        )
        job = _make_job("svc-abc123", "svc")

        manager = GitManager(push_timeout_s=30)
        result = await manager.commit_and_push(job, target, working_copy)

        # Succeeded outcome: pushed status carrying the derived branch name.
        expected_branch = "agent/svc-abc123"
        assert result.status is PushStatus.PUSHED
        assert result.branch_name == expected_branch
        assert result.failure_reason is None

        # The branch genuinely exists on the remote with the committed change.
        remote_repo = Repo(remote_dir)
        remote_branches = {head.name for head in remote_repo.heads}
        assert expected_branch in remote_branches

        pushed_commit = remote_repo.heads[expected_branch].commit
        assert job.id in pushed_commit.message
        # The new file is present in the pushed tree.
        assert "feature.txt" in pushed_commit.tree


async def test_push_failure_maps_to_push_error_reason(monkeypatch):
    """An unreachable remote makes the push fail and map to push-error (5.5).

    Everything up to the push succeeds (branch created, commit made), but the
    remote URL points nowhere reachable, so the push cannot complete. The result
    is a Failed outcome carrying :data:`PUSH_ERROR_REASON` and no branch name.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work_dir = root / "work"

        repo = _init_work_repo(work_dir)
        working_copy = Path(repo.working_tree_dir)

        # Dirty working tree so the call reaches the push step.
        (working_copy / "feature.txt").write_text("a generated change\n", encoding="utf-8")

        creds_env = f"GIT_TOKEN_{uuid.uuid4().hex.upper()}"
        monkeypatch.setenv(creds_env, "unused-but-present-token")

        # A remote path that does not exist: the push has nowhere to go.
        bad_remote = root / "does-not-exist.git"

        target = RegisteredTarget(
            name="svc",
            directory_path=working_copy,
            repo_remote=str(bad_remote),
            default_branch="main",
            branch_scheme="agent/{job_id}",
            credentials_ref=creds_env,
        )
        job = _make_job("svc-def456", "svc")

        manager = GitManager(push_timeout_s=30)
        result = await manager.commit_and_push(job, target, working_copy)

        # The push failed and is reported as a push error, with no branch name.
        assert result.status is PushStatus.FAILED
        assert result.failure_reason == PUSH_ERROR_REASON
        assert result.branch_name is None
