"""Property-based test for the commit/push no-changes outcome (task 7.3).

Covers Property 18: a Job whose working copy has no differences from its synced
default-branch state transitions to Succeeded with a no-changes note, and the
commit and push steps are skipped.

The working copy is a real, temp-dir local Git repository whose working tree is
clean (every committed file is present and unmodified, no staged changes, no
untracked files). For any such clean copy -- across varying committed file sets,
target names, branch schemes, Job identifiers, and credentials references --
:meth:`GitManager.commit_and_push` must return :attr:`PushStatus.NO_CHANGES`
with a human-readable ``result_note`` and ``None`` ``branch_name`` (no push),
and the repository's commit history must be left untouched.

**Validates: Requirements 5.6**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from git import Repo
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.git_manager import GitManager, PushStatus
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# A single legal path component: lowercase letters/digits only, so a drawn
# segment is always a valid file name and never collides with git internals.
_SAFE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=12,
)

# A non-empty mapping of relative file paths -> text contents to commit as the
# repository's clean baseline. At least one file guarantees a real commit.
_FILE_SETS = st.dictionaries(
    keys=_SAFE_NAME,
    values=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
        max_size=80,
    ),
    min_size=1,
    max_size=5,
)


def _make_target(name: str, branch_scheme: str, credentials_ref: str) -> RegisteredTarget:
    """Build a RegisteredTarget; remote is irrelevant on the no-changes path."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("/workspace") / name,
        repo_remote="https://example.invalid/repo.git",
        branch_scheme=branch_scheme,
        credentials_ref=credentials_ref,
    )


def _make_job(job_id: str, target_name: str) -> Job:
    return Job(
        id=job_id,
        target_name=target_name,
        idea="no-op idea",
        status=JobStatus.RUNNING,
        submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        channel_id=1,
        user_id=2,
    )


def _init_clean_repo(root: Path, files: dict[str, str]) -> Repo:
    """Initialize a local git repo at ``root`` with a committed, clean tree."""
    repo = Repo.init(root)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "agent@example.invalid")
        cw.set_value("user", "name", "Agent Test")
    for rel_name, contents in files.items():
        (root / rel_name).write_text(contents, encoding="utf-8")
    repo.git.add("--all")
    repo.index.commit("baseline")
    # Sanity: the working copy must be clean (no diff from the synced state).
    assert not repo.is_dirty(index=True, working_tree=True, untracked_files=True)
    return repo


# Feature: discord-ollama-coding-agent, Property 18
# Property 18: No file changes yields Succeeded with a no-changes note and no push.
# Validates: Requirements 5.6
@settings(max_examples=20)
@given(
    files=_FILE_SETS,
    target_name=_SAFE_NAME,
    branch_scheme=st.sampled_from(["{job_id}", "agent/{job_id}", "{target_name}/{job_id}"]),
    credentials_ref=st.sampled_from(["GIT_TOKEN_ENV", "MISSING_TOKEN_ENV", "/no/such/secret"]),
)
@pytest.mark.asyncio
async def test_no_changes_yields_succeeded_no_changes_with_no_push(
    files: dict[str, str],
    target_name: str,
    branch_scheme: str,
    credentials_ref: str,
):
    """A clean working copy yields NO_CHANGES with a note and no push.

    The commit/push steps are skipped: no new commit is created and no branch
    name is returned (nothing was pushed). The outcome is independent of the
    credentials reference because the no-changes check precedes credential
    resolution.
    """
    with tempfile.TemporaryDirectory() as root_name:
        working_copy = Path(root_name)
        repo = _init_clean_repo(working_copy, files)
        head_before = repo.head.commit.hexsha
        commit_count_before = sum(1 for _ in repo.iter_commits())

        target = _make_target(target_name, branch_scheme, credentials_ref)
        job = _make_job(f"{target_name}-job1", target_name)

        manager = GitManager()
        result = await manager.commit_and_push(job, target, working_copy)

        # Succeeded outcome: no-changes status carrying a human-readable note.
        assert result.status is PushStatus.NO_CHANGES
        assert result.result_note is not None and result.result_note.strip()
        # No push happened: no branch name and no failure reason.
        assert result.branch_name is None
        assert result.failure_reason is None

        # Commit step skipped: history is exactly as it was before the call.
        repo_after = Repo(working_copy)
        assert repo_after.head.commit.hexsha == head_before
        assert sum(1 for _ in repo_after.iter_commits()) == commit_count_before
        assert not repo_after.is_dirty(
            index=True, working_tree=True, untracked_files=True
        )
