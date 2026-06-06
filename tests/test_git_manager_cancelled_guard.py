"""Property-based test for the Cancelled-Job commit/push guard (task 7.5).

Covers Property 16: Cancelled Jobs never stage, commit, or push.

*For any* Job that is in status Cancelled, :meth:`GitManager.commit_and_push`
performs **no** staging, commit, or push for that Job, regardless of any files
produced before cancellation. The call returns :attr:`PushStatus.CANCELLED`,
creates no new commit in the local working copy, and leaves the remote's refs
exactly as they were (Req 5.9, 8.4, 8.5).

The test exercises the real :class:`GitManager` against a temp-directory local
Git repository whose working tree is genuinely dirty (changes are present that
would otherwise be committed and pushed), wired to a local bare ``origin``
remote so that "no push attempted" is verified directly: the bare remote's refs
are snapshotted before the call and asserted unchanged afterwards. Crucially the
target's credentials reference *resolves to a real value* and the working tree
*has changes*, so the only thing that can prevent a commit/push is the Cancelled
guard itself -- pinning the property across varying target names, branch
schemes, credentials forms, working-copy changes, and Job identifiers.

**Validates: Requirements 5.9, 8.4, 8.5**
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.git_manager import GitManager, PushStatus
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# Safe single-path-component fragments: lowercase letters and digits only, so a
# drawn segment is always a valid file name / target name and never collides
# with git internals.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12
)

# Branch-naming schemes that all derive a valid branch name from the Job id.
_BRANCH_SCHEMES = st.sampled_from(
    ["{job_id}", "agent/{job_id}", "{target_name}/{job_id}", "feat-{job_id}"]
)

# The kinds of pending change a Cancelled Job might have produced before it was
# cancelled. Each makes the working tree genuinely dirty so that, absent the
# guard, there would be something to stage, commit, and push.
_CHANGE_KIND = st.sampled_from(["untracked", "modified", "deleted", "staged"])

# Whether the target's credentials resolve to a real value. Even when they do
# (so the missing-credentials guard would NOT trip), a Cancelled Job must still
# never commit or push.
_CREDS_RESOLVE = st.booleans()


def _apply_pending_change(work_dir: Path, repo: Repo, change_kind: str) -> None:
    """Dirty the working tree in the manner described by ``change_kind``."""
    tracked = work_dir / "README.md"
    if change_kind == "untracked":
        (work_dir / "feature.txt").write_text("a generated change\n", encoding="utf-8")
    elif change_kind == "modified":
        tracked.write_text("seed\nmodified by the agent\n", encoding="utf-8")
    elif change_kind == "deleted":
        tracked.unlink()
    else:  # staged: a new file added to the index but not yet committed.
        staged = work_dir / "staged.txt"
        staged.write_text("staged change\n", encoding="utf-8")
        repo.git.add("staged.txt")


def _init_dirty_repo_with_remote(
    root: Path, change_kind: str
) -> tuple[Repo, Path, dict[str, str]]:
    """Create a dirty work-copy repo plus a local bare ``origin`` remote.

    Returns the work-copy repo, the bare remote path, and a snapshot of the
    remote's refs (ref path -> commit hexsha) taken *before* any push could have
    happened, so the caller can assert it is unchanged.
    """
    work_dir = root / "work"
    remote_dir = root / "remote.git"
    work_dir.mkdir(parents=True)
    remote_dir.mkdir(parents=True)

    # A real local bare repo serves as the push target.
    Repo.init(remote_dir, bare=True)

    repo = Repo.init(work_dir)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test Agent")
        cw.set_value("user", "email", "agent@example.invalid")

    # Establish an initial committed baseline and publish it to the remote so
    # the remote starts with a known set of refs.
    (work_dir / "README.md").write_text("seed\n", encoding="utf-8")
    repo.git.add("--all")
    repo.index.commit("initial commit")
    origin = repo.create_remote("origin", str(remote_dir))
    origin.push(refspec="HEAD:refs/heads/main")

    # Snapshot the remote refs after the baseline push but before commit_and_push.
    remote_repo = Repo(remote_dir)
    refs_before = {ref.path: ref.commit.hexsha for ref in remote_repo.refs}

    # Produce the pending change so, absent the guard, there would be something
    # to stage, commit, and push.
    _apply_pending_change(work_dir, repo, change_kind)
    assert repo.is_dirty(index=True, working_tree=True, untracked_files=True)

    return repo, remote_dir, refs_before


# Feature: discord-ollama-coding-agent, Property 16: Cancelled Jobs never stage, commit, or push
# Validates: Requirements 5.9, 8.4, 8.5
@settings(max_examples=20)
@given(
    target_name=_SAFE_SEGMENT,
    branch_scheme=_BRANCH_SCHEMES,
    change_kind=_CHANGE_KIND,
    creds_resolve=_CREDS_RESOLVE,
    idea=st.text(min_size=0, max_size=40),
)
def test_cancelled_job_never_stages_commits_or_pushes(
    target_name: str,
    branch_scheme: str,
    change_kind: str,
    creds_resolve: bool,
    idea: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo, remote_dir, refs_before = _init_dirty_repo_with_remote(root, change_kind)
        working_copy = Path(repo.working_tree_dir)

        head_before = repo.head.commit.hexsha
        commit_count_before = sum(1 for _ in repo.iter_commits())
        # The pre-call dirty status, so we can prove the guard touched nothing
        # (no staging) afterwards.
        status_before = repo.git.status("--porcelain")

        # Resolvable credentials when requested, so the *only* thing that can
        # prevent a commit/push is the Cancelled guard -- not missing creds.
        if creds_resolve:
            creds_env = f"GIT_TOKEN_{uuid.uuid4().hex.upper()}"
            os.environ[creds_env] = "unused-but-present-token"
            credentials_ref = creds_env
        else:
            credentials_ref = f"GIT_TOKEN_ABSENT_{uuid.uuid4().hex.upper()}"
            os.environ.pop(credentials_ref, None)

        try:
            target = RegisteredTarget(
                name=target_name,
                directory_path=working_copy,
                repo_remote=str(remote_dir),
                default_branch="main",
                branch_scheme=branch_scheme,
                credentials_ref=credentials_ref,
            )
            job = Job(
                id=f"{target_name}-{uuid.uuid4().hex[:8]}",
                target_name=target_name,
                idea=idea,
                status=JobStatus.CANCELLED,
                submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                channel_id=99,
                user_id=7,
            )

            manager = GitManager(push_timeout_s=30)
            result = asyncio.run(manager.commit_and_push(job, target, working_copy))
        finally:
            if creds_resolve:
                os.environ.pop(credentials_ref, None)

        # The Git_Manager refused to act: Cancelled outcome, no branch name, no
        # failure reason (it never reached commit/push).
        assert result.status is PushStatus.CANCELLED
        assert result.branch_name is None
        assert result.failure_reason is None

        # No commit was created: local history is exactly as it was before.
        repo_after = Repo(working_copy)
        assert repo_after.head.commit.hexsha == head_before
        assert sum(1 for _ in repo_after.iter_commits()) == commit_count_before

        # No staging happened: the working-tree status is untouched. (In
        # particular the guard did not run `git add --all`.)
        assert repo_after.git.status("--porcelain") == status_before

        # No push happened: the remote's refs are exactly as they were before
        # the call (no new branch, no advanced ref).
        remote_repo = Repo(remote_dir)
        refs_after = {ref.path: ref.commit.hexsha for ref in remote_repo.refs}
        assert refs_after == refs_before
