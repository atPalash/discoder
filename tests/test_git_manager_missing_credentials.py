"""Property-based test for the missing-credentials push guard (task 7.4).

Covers Property 17: Missing credentials prevent any push.

*For any* Job whose Registered_Target's configured Git credentials reference
resolves to no credentials, :meth:`GitManager.commit_and_push` returns a Failed
result carrying the :data:`MISSING_CREDENTIALS_REASON` reason and performs **no
push** -- even though the working copy has changes that would otherwise be
committed and pushed (Req 5.7).

The test exercises the real :class:`GitManager` against a temp-directory local
Git repository whose working tree is genuinely dirty, with an ``origin`` remote
wired to a local bare repository so that "no push attempted" can be verified
directly: the bare remote's refs are snapshotted before the call and asserted
unchanged afterwards. The credentials reference is generated in each of the
distinct forms that :func:`resolve_credentials` maps to *no value* (empty,
whitespace-only, an unset environment-variable name, and a non-existent secret
path), so the property is pinned across the whole "resolves to nothing" input
space.

**Validates: Requirements 5.7**
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from hypothesis import given
from hypothesis import strategies as st

from discord_ollama_agent.git_manager import (
    MISSING_CREDENTIALS_REASON,
    GitManager,
    PushStatus,
)
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# Job statuses for which commit_and_push proceeds far enough to reach the
# credentials guard. Cancelled is excluded: a Cancelled Job is refused before
# the credentials check (Req 5.9, Property 16) and is covered by its own test.
_NON_CANCELLED_STATUSES = st.sampled_from(
    [
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
    ]
)

# Safe single-path-component fragments: lowercase letters and digits only.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12
)

# Branch-naming schemes that all derive a valid branch name from the Job id.
_BRANCH_SCHEMES = st.sampled_from(
    ["{job_id}", "agent/{job_id}", "{target_name}/{job_id}", "feat-{job_id}"]
)

# A credentials reference that resolves_credentials() maps to *no value*.
#
#   - "" / whitespace-only -> rejected up front as empty
#   - an UNSET environment-variable name -> env lookup yields nothing
#   - a non-existent filesystem path -> file lookup yields nothing
#
# The env-var/path forms are made unique per example so they cannot collide with
# anything actually set in the test environment.
_CREDS_FORM = st.sampled_from(["empty", "whitespace", "unset_env", "missing_path"])


def _make_unresolvable_credentials_ref(form: str) -> str:
    """Build a credentials_ref guaranteed to resolve to no value."""
    if form == "empty":
        return ""
    if form == "whitespace":
        return "   \t  "
    if form == "unset_env":
        name = f"GIT_TOKEN_ABSENT_{uuid.uuid4().hex.upper()}"
        # Ensure it is genuinely unset so resolution yields nothing.
        os.environ.pop(name, None)
        return name
    # missing_path: a path that does not exist on disk.
    return str(Path(tempfile.gettempdir()) / f"no-such-secret-{uuid.uuid4().hex}")


def _init_dirty_repo_with_remote(root: Path) -> tuple[Repo, Path, dict[str, str]]:
    """Create a work-copy repo (dirty) plus a local bare ``origin`` remote.

    Returns the work-copy repo, the bare remote path, and a snapshot of the
    remote's refs (ref name -> commit hexsha) taken *before* any push could have
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

    # Make the working tree genuinely dirty so, absent the guard, there would be
    # something to commit and push.
    (work_dir / "feature.txt").write_text("a generated change\n", encoding="utf-8")

    return repo, remote_dir, refs_before


# Feature: discord-ollama-coding-agent, Property 17: Missing credentials prevent any push
# Validates: Requirements 5.7
@given(
    target_name=_SAFE_SEGMENT,
    branch_scheme=_BRANCH_SCHEMES,
    creds_form=_CREDS_FORM,
    status=_NON_CANCELLED_STATUSES,
    idea=st.text(min_size=0, max_size=40),
)
def test_missing_credentials_prevent_any_push(
    target_name: str,
    branch_scheme: str,
    creds_form: str,
    status: JobStatus,
    idea: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo, remote_dir, refs_before = _init_dirty_repo_with_remote(root)
        working_copy = Path(repo.working_tree_dir)

        credentials_ref = _make_unresolvable_credentials_ref(creds_form)
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
            status=status,
            submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            channel_id=99,
            user_id=7,
        )

        manager = GitManager(push_timeout_s=30)
        result = asyncio.run(manager.commit_and_push(job, target, working_copy))

        # The Job fails specifically with the missing-credentials reason ...
        assert result.status is PushStatus.FAILED
        assert result.failure_reason == MISSING_CREDENTIALS_REASON
        assert result.branch_name is None

        # ... and no push was attempted: the remote's refs are exactly as they
        # were before the call (no new branch, no advanced ref).
        remote_repo = Repo(remote_dir)
        refs_after = {ref.path: ref.commit.hexsha for ref in remote_repo.refs}
        assert refs_after == refs_before
