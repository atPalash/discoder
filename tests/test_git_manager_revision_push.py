"""Integration test for GitManager.commit_and_push_revision (task 7.8).

Exercises the full revision push flow against a real, local fixture remote (a
bare Git repository on disk): a base branch is first created and pushed via
``commit_and_push``; then new changes are produced and ``commit_and_push_revision``
is invoked with a Job carrying that branch name. The test asserts the new
commits land on the *same* branch on the fixture remote.

Validates: Requirements 11.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from git import Repo

from discord_ollama_agent.git_manager import GitManager, PushStatus
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

CREDENTIALS_ENV_VAR = "GIT_MANAGER_REVISION_TEST_TOKEN"
DEFAULT_BRANCH = "main"


def _make_job(**overrides) -> Job:
    base = dict(
        id="svc-rev-1",
        target_name="svc",
        idea="add a feature",
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=42,
        user_id=7,
        status=JobStatus.RUNNING,
    )
    base.update(overrides)
    return Job(**base)


def _init_fixture_remote(remote_dir: Path) -> str:
    """Create a bare fixture remote seeded with an initial commit on ``main``.

    Returns the file:// URL GitManager will treat as the target's remote.
    """
    bare = Repo.init(remote_dir, bare=True, initial_branch=DEFAULT_BRANCH)

    # Seed the bare remote with one commit on the default branch by pushing from
    # a throwaway working copy, so a later clone/sync of the default branch works.
    seed_dir = remote_dir.parent / "seed"
    seed = Repo.init(seed_dir, initial_branch=DEFAULT_BRANCH)
    with seed.config_writer() as cw:
        cw.set_value("user", "name", "Seed User")
        cw.set_value("user", "email", "seed@example.com")
    (seed_dir / "README.md").write_text("seed\n", encoding="utf-8")
    seed.git.add("--all")
    seed.index.commit("seed commit")
    seed.create_remote("origin", str(remote_dir))
    seed.remote("origin").push(refspec=f"{DEFAULT_BRANCH}:{DEFAULT_BRANCH}")

    assert DEFAULT_BRANCH in {h.name for h in bare.heads}
    return str(remote_dir)


def _commits_on_remote_branch(remote_dir: Path, branch: str) -> list[str]:
    """Return commit messages (newest first) for ``branch`` on the fixture remote."""
    bare = Repo(remote_dir)
    return [c.message.strip() for c in bare.iter_commits(branch)]


@pytest.fixture()
def credentials_env(monkeypatch):
    """Make the target's credentials_ref resolve to a non-empty value."""
    monkeypatch.setenv(CREDENTIALS_ENV_VAR, "fixture-token")
    return CREDENTIALS_ENV_VAR


async def test_revision_adds_new_commits_to_existing_branch_on_remote(
    tmp_path: Path, credentials_env: str
):
    """A revision pushes new commits onto the base Job's existing branch (Req 11.4)."""
    remote_dir = tmp_path / "remote.git"
    remote_url = _init_fixture_remote(remote_dir)

    target = RegisteredTarget(
        name="svc",
        directory_path=tmp_path / "workspace" / "svc",
        repo_remote=remote_url,
        default_branch=DEFAULT_BRANCH,
        branch_scheme="agent/{job_id}",
        credentials_ref=credentials_env,
    )

    manager = GitManager(push_timeout_s=60)
    working_copy = tmp_path / "working_copy"

    # Sync the isolated working copy to the fixture remote's default branch.
    sync = await manager.sync_default_branch(target, working_copy)
    assert sync.ok, sync.detail

    # --- Base branch: produce a change and push it via commit_and_push. ---
    base_job = _make_job(id="svc-rev-1")
    (working_copy / "feature.txt").write_text("version 1\n", encoding="utf-8")

    base_result = await manager.commit_and_push(base_job, target, working_copy)
    assert base_result.status is PushStatus.PUSHED, base_result.detail
    branch_name = base_result.branch_name
    assert branch_name == "agent/svc-rev-1"

    base_commits = _commits_on_remote_branch(remote_dir, branch_name)
    assert base_commits == [f"Apply changes for Job {base_job.id}", "seed commit"]

    # --- Revision: produce further changes and push to the SAME branch. ---
    revision_job = _make_job(
        id="svc-rev-1-r1",
        is_revision=True,
        base_job_id=base_job.id,
        branch_name=branch_name,
    )
    (working_copy / "feature.txt").write_text("version 2\n", encoding="utf-8")
    (working_copy / "extra.txt").write_text("more\n", encoding="utf-8")

    revision_result = await manager.commit_and_push_revision(
        revision_job, target, working_copy
    )

    # The revision pushed to the same branch the base Job created.
    assert revision_result.status is PushStatus.PUSHED, revision_result.detail
    assert revision_result.branch_name == branch_name

    # The fixture remote's branch now carries the revision's new commit on top of
    # the base commit (no new branch was created).
    remote_branches = {h.name for h in Repo(remote_dir).heads}
    assert remote_branches == {DEFAULT_BRANCH, branch_name}

    revised_commits = _commits_on_remote_branch(remote_dir, branch_name)
    assert revised_commits == [
        f"Apply changes for Job {revision_job.id}",
        f"Apply changes for Job {base_job.id}",
        "seed commit",
    ]
    assert len(revised_commits) == len(base_commits) + 1

    # The newest commit on the remote branch reflects the revision's content.
    bare = Repo(remote_dir)
    tip = next(bare.iter_commits(branch_name))
    tree_blobs = {blob.name for blob in tip.tree.blobs}
    assert {"feature.txt", "extra.txt", "README.md"} <= tree_blobs
    assert tip.tree["feature.txt"].data_stream.read().decode() == "version 2\n"
