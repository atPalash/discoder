"""Integration tests for default-branch sync (task 7.6).

Exercises :meth:`GitManager.sync_default_branch` end-to-end against a real,
local fixture Git remote (a bare repository on disk -- no network), covering:

- Req 4.12: syncing a Job's working copy updates it to the latest state of the
  target's configured default branch obtained from the target's remote. Both the
  fresh-clone path (no working copy yet) and the fetch+reset path (an existing,
  stale working copy) are covered, and the synced files/HEAD are asserted to
  match the remote's default-branch tip.
- Req 4.13: a fetch/reset failure (here, an unreachable/nonexistent remote) maps
  to a :data:`SYNC_FAILURE_REASON` result rather than raising.

These are integration tests: they drive the real GitPython code paths against an
actual on-disk bare remote built by ``git`` itself, so a passing assertion proves
the sync genuinely moved bytes between repositories.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from git import Repo

from discord_ollama_agent.git_manager import (
    SYNC_FAILURE_REASON,
    GitManager,
)
from discord_ollama_agent.models.target import RegisteredTarget


def _configure_identity(repo: Repo) -> None:
    """Give ``repo`` a committer identity so commits succeed in any environment."""
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "agent@example.invalid")
        cw.set_value("user", "name", "Agent Test")


def _make_fixture_remote(
    root: Path, default_branch: str, files: dict[str, str]
) -> str:
    """Create a local bare remote whose ``default_branch`` holds ``files``.

    Builds a normal "author" repo, commits ``files`` on ``default_branch``, then
    pushes it into a sibling bare repository that serves as the fixture remote.
    Returns the bare repo's filesystem path (a valid local Git remote URL).
    """
    bare_path = root / "remote.git"
    Repo.init(bare_path, bare=True)

    author_path = root / "author"
    author = Repo.init(author_path)
    _configure_identity(author)
    # Name the initial branch deterministically regardless of git's default.
    author.git.checkout("-B", default_branch)
    for rel_name, contents in files.items():
        (author_path / rel_name).write_text(contents, encoding="utf-8")
    author.git.add("--all")
    author.index.commit("seed default branch")
    author.create_remote("origin", str(bare_path))
    author.git.push("origin", f"{default_branch}:{default_branch}")

    return str(bare_path)


def _make_target(name: str, repo_remote: str, default_branch: str) -> RegisteredTarget:
    """Build a RegisteredTarget pointing at the fixture remote (no credentials)."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("/workspace") / name,
        repo_remote=repo_remote,
        default_branch=default_branch,
        # The local fixture remote is public on disk, so an unresolved
        # credentials reference is fine; only its absence is observable.
        credentials_ref="UNUSED_CREDENTIALS_ENV",
    )


@pytest.mark.asyncio
async def test_sync_clones_fresh_working_copy_to_default_branch():
    """Req 4.12: syncing an empty working copy clones the remote's default branch."""
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        files = {"README.md": "hello from default", "src.py": "print('hi')\n"}
        remote_url = _make_fixture_remote(root, "main", files)

        target = _make_target("alpha", remote_url, "main")
        working_copy = root / "work"  # does not exist yet -> clone path

        manager = GitManager()
        result = await manager.sync_default_branch(target, working_copy)

        assert result.ok is True
        assert result.failure_reason is None

        # The working copy now contains exactly the default-branch files.
        for rel_name, contents in files.items():
            assert (working_copy / rel_name).read_text(encoding="utf-8") == contents

        # HEAD matches the remote default-branch tip.
        synced = Repo(working_copy)
        remote_tip = Repo(remote_url).commit("main").hexsha
        assert synced.head.commit.hexsha == remote_tip


@pytest.mark.asyncio
async def test_sync_updates_existing_working_copy_to_latest_default_branch():
    """Req 4.12: a stale existing working copy is fetched + reset to the latest tip."""
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        initial = {"README.md": "v1"}
        remote_url = _make_fixture_remote(root, "main", initial)
        target = _make_target("beta", remote_url, "main")

        working_copy = root / "work"
        manager = GitManager()

        # First sync: clones the initial state.
        first = await manager.sync_default_branch(target, working_copy)
        assert first.ok is True
        assert (working_copy / "README.md").read_text(encoding="utf-8") == "v1"

        # Advance the remote default branch with a new commit via the author repo.
        author = Repo(root / "author")
        (root / "author" / "README.md").write_text("v2", encoding="utf-8")
        (root / "author" / "feature.txt").write_text("new file", encoding="utf-8")
        author.git.add("--all")
        author.index.commit("advance default branch")
        author.git.push("origin", "main:main")
        new_tip = author.commit("main").hexsha

        # Dirty the local working copy to prove the reset is a hard reset.
        (working_copy / "README.md").write_text("local junk", encoding="utf-8")
        (working_copy / "untracked.txt").write_text("scratch", encoding="utf-8")

        # Second sync: fetch + hard reset to the advanced tip.
        second = await manager.sync_default_branch(target, working_copy)
        assert second.ok is True
        assert second.failure_reason is None

        assert (working_copy / "README.md").read_text(encoding="utf-8") == "v2"
        assert (working_copy / "feature.txt").read_text(encoding="utf-8") == "new file"

        synced = Repo(working_copy)
        assert synced.head.commit.hexsha == new_tip
        # The hard reset discarded the local modification to the tracked file.
        assert not synced.is_dirty(index=True, working_tree=True)


@pytest.mark.asyncio
async def test_sync_from_unreachable_remote_maps_to_sync_failure():
    """Req 4.13: a failing fetch/clone maps to a SYNC_FAILURE_REASON result."""
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        missing_remote = str(root / "does-not-exist.git")
        target = _make_target("gamma", missing_remote, "main")
        working_copy = root / "work"

        manager = GitManager()
        result = await manager.sync_default_branch(target, working_copy)

        assert result.ok is False
        assert result.failure_reason == SYNC_FAILURE_REASON
        # A secret-free detail is surfaced for the operator log.
        assert result.detail is not None and result.detail.strip()


@pytest.mark.asyncio
async def test_sync_missing_default_branch_maps_to_sync_failure():
    """Req 4.13: fetching a default branch absent on the remote is a sync failure."""
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        # Remote only has 'main'; target asks for a branch that doesn't exist.
        remote_url = _make_fixture_remote(root, "main", {"README.md": "hi"})
        target = _make_target("delta", remote_url, "release")

        # Pre-create the working copy as a clone of the remote so the fetch path
        # (not the clone path) is exercised, then ask for the missing branch.
        working_copy = root / "work"
        Repo.clone_from(remote_url, str(working_copy), branch="main")

        manager = GitManager()
        result = await manager.sync_default_branch(target, working_copy)

        assert result.ok is False
        assert result.failure_reason == SYNC_FAILURE_REASON
        assert result.detail is not None and result.detail.strip()
