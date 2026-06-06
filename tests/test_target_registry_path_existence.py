"""Example tests for registration path-existence rejection (task 4.7).

Covers Req 10.4: ``/addtarget`` must reject a directory path that does not refer
to an *existing directory*. ``TargetRegistry.add`` enforces this by resolving the
candidate path (after the workspace-containment check passes) and raising
:class:`TargetDirectoryNotFoundError` unless the resolved path is an existing
directory.

These are example-based tests pinning the two concrete ways a path can fail the
existence check while still resolving inside the workspace:

* a path that does not exist at all, and
* a path that exists but is a regular file rather than a directory.

Each case asserts the registration is rejected with
:class:`TargetDirectoryNotFoundError` and that the registry -- both its in-memory
state and its backing file -- is left untouched.

**Validates: Requirements 10.4**
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from discord_ollama_agent.errors import TargetDirectoryNotFoundError
from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.target_registry import TargetRegistry


def _make_target(name: str, directory_path: Path) -> RegisteredTarget:
    """Build a RegisteredTarget that is valid except for its directory path.

    Every field other than ``directory_path`` is held at a fixed valid value so
    the only possible registration failure is the path-existence check.
    """
    return RegisteredTarget(
        name=name,
        directory_path=directory_path,
        repo_remote="https://example.invalid/repo.git",
        credentials_ref="GIT_TOKEN_ENV",
    )


def _persisted_targets(registry_path: Path) -> list[dict]:
    """Return the persisted target entries, or ``[]`` if the file is absent."""
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return data.get("targets", []) if isinstance(data, dict) else data


# Validates: Requirements 10.4
def test_add_rejects_nonexistent_path():
    """A path inside the workspace that does not exist is rejected (Req 10.4).

    The candidate lives lexically inside the workspace (so the containment check
    passes) but no such directory was ever created, so ``add`` must raise
    :class:`TargetDirectoryNotFoundError` and leave the registry empty and the
    backing file unwritten.
    """
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        workspace = root / "workspace"
        workspace.mkdir()
        registry_path = root / "config" / "registry.json"

        registry = TargetRegistry()
        registry.load(registry_path, workspace)
        assert registry.names() == []

        missing = workspace / "does_not_exist"
        assert not missing.exists()
        target = _make_target("ghost", missing)

        with pytest.raises(TargetDirectoryNotFoundError):
            registry.add(target)

        # Registry state and backing file are untouched by the rejection.
        assert not registry.exists("ghost")
        assert registry.get("ghost") is None
        assert registry.names() == []
        assert _persisted_targets(registry_path) == []


# Validates: Requirements 10.4
def test_add_rejects_path_that_is_a_file_not_a_directory():
    """A path that resolves to a regular file (not a directory) is rejected (Req 10.4).

    The candidate exists inside the workspace, so both the containment check and
    "the path exists" hold, but it is a file rather than a directory. ``add``
    must still raise :class:`TargetDirectoryNotFoundError` and leave the registry
    and its backing file unchanged.
    """
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        workspace = root / "workspace"
        workspace.mkdir()
        registry_path = root / "config" / "registry.json"

        registry = TargetRegistry()
        registry.load(registry_path, workspace)
        assert registry.names() == []

        file_path = workspace / "a_file.txt"
        file_path.write_text("not a directory", encoding="utf-8")
        assert file_path.is_file()
        target = _make_target("filish", file_path)

        with pytest.raises(TargetDirectoryNotFoundError):
            registry.add(target)

        # Registry state and backing file are untouched by the rejection.
        assert not registry.exists("filish")
        assert registry.get("filish") is None
        assert registry.names() == []
        assert _persisted_targets(registry_path) == []
