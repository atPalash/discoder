"""Property-based test for branch/commit identifier references (task 7.2).

Covers Property 19: for any Job, the branch name derived from the Job identifier
via the target's branch-naming scheme contains the associated target's name, and
the commit message created for that Job references the Job identifier.

Because a Job identifier has the shape ``"{target_name}-{suffix}"`` (Req 7.1),
formatting the target's ``branch_scheme`` with the Job id always yields a branch
name that includes the target name (Req 5.1). The commit-message half of the
property is exercised end to end against a real, temp-dir local Git repository
with a local bare remote: ``commit_and_push`` stages a change, commits, and
pushes, and the resulting commit message must reference the Job id (Req 5.2).

**Validates: Requirements 5.1, 5.2**
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from git import Repo
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.git_manager import GitManager, PushStatus
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# A single legal identifier segment: lowercase letters/digits only, so a drawn
# segment is always a valid target name, branch component, and file name.
_SAFE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=16,
)

# Branch-naming schemes that all reference the Job id (and therefore, since the
# Job id contains the target name, the target name too); one also references the
# target name explicitly.
_BRANCH_SCHEMES = st.sampled_from(
    ["{job_id}", "agent/{job_id}", "{target_name}/{job_id}", "feature/{job_id}"]
)


def _make_target(name: str, branch_scheme: str, repo_remote: str, credentials_ref: str) -> RegisteredTarget:
    return RegisteredTarget(
        name=name,
        directory_path=Path("/workspace") / name,
        repo_remote=repo_remote,
        default_branch="main",
        branch_scheme=branch_scheme,
        credentials_ref=credentials_ref,
    )


def _make_job(job_id: str, target_name: str) -> Job:
    return Job(
        id=job_id,
        target_name=target_name,
        idea="add a feature",
        status=JobStatus.RUNNING,
        submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        channel_id=1,
        user_id=2,
    )


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


# Feature: discord-ollama-coding-agent, Property 19
# Property 19: Branch and commit reference the target and Job identifier.
# Validates: Requirements 5.1, 5.2
@settings(max_examples=20)
@given(
    target_name=_SAFE_NAME,
    suffix=_SAFE_NAME,
    branch_scheme=_BRANCH_SCHEMES,
)
@pytest.mark.asyncio
async def test_branch_contains_target_and_commit_references_job_id(
    target_name: str,
    suffix: str,
    branch_scheme: str,
):
    """Derived branch contains the target name; the commit references the Job id.

    The Job id is ``"{target_name}-{suffix}"`` (Req 7.1), so the branch derived
    via the target's scheme contains the target name (Req 5.1). Pushing a real
    change produces a commit whose message references the Job id (Req 5.2).
    """
    job_id = f"{target_name}-{suffix}"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        work_dir = root / "work"
        remote_dir = root / "remote.git"

        # A real local bare repository serves as the push target.
        Repo.init(remote_dir, bare=True)
        repo = _init_work_repo(work_dir)
        working_copy = Path(repo.working_tree_dir)

        # A resolvable (locally unused) credentials reference clears the guard.
        creds_env = f"GIT_TOKEN_{uuid.uuid4().hex.upper()}"
        os.environ[creds_env] = "unused-but-present-token"
        try:
            target = _make_target(target_name, branch_scheme, str(remote_dir), creds_env)
            job = _make_job(job_id, target_name)

            manager = GitManager(push_timeout_s=30)

            # Part 1: the derived branch name contains the target name (Req 5.1).
            branch_name = manager.branch_name_for(job, target)
            assert target_name in branch_name
            assert job_id in branch_name

            # Part 2: pushing a change yields a commit referencing the Job id.
            (working_copy / "feature.txt").write_text("a change\n", encoding="utf-8")
            result = await manager.commit_and_push(job, target, working_copy)

            assert result.status is PushStatus.PUSHED
            assert result.branch_name == branch_name

            # The pushed commit's message references the Job identifier (Req 5.2).
            remote_repo = Repo(remote_dir)
            pushed_commit = remote_repo.heads[branch_name].commit
            assert job.id in pushed_commit.message
        finally:
            os.environ.pop(creds_env, None)
