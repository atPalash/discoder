"""Property-based test for workspace-confined target registration (task 4.2).

Covers Property 12: target registration is confined to the Workspace_Directory.

*For any* candidate directory path supplied to ``TargetRegistry.add``, the
target is registered **if and only if** the path resolves to a location inside
the ``Workspace_Directory``. Paths escaping the workspace -- via absolute paths
outside it, parent-directory (``..``) traversal, or symbolic links that resolve
outside -- are rejected with :class:`PathOutsideWorkspaceError` and leave the
registry unchanged. In-workspace paths (including those reached through a
symlink that resolves back inside) are accepted.

To isolate the containment property (Req 10.3) from the separate
directory-existence requirement (Req 10.4), every in-workspace candidate in this
test is a real, existing directory, so "resolves inside the workspace" lines up
exactly with "is accepted".

**Validates: Requirements 10.3**
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.errors import PathOutsideWorkspaceError
from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.target_registry import TargetRegistry

# Safe path/name segments: lowercase letters and digits only, so a drawn segment
# is always a legal single path component and never ``.`` or ``..``.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=8,
)

# The kinds of candidate path we build. ``inside_*`` categories genuinely live
# inside the workspace; ``escape_*`` categories genuinely resolve outside it.
# The expected-inside oracle is derived from the category alone (test knowledge),
# never by re-running the implementation's own resolve logic.
_INSIDE_CATEGORIES = ("inside_relative", "inside_root", "inside_symlink")
_ESCAPE_CATEGORIES = ("escape_dotdot", "escape_absolute", "escape_symlink")
_CATEGORIES = _INSIDE_CATEGORIES + _ESCAPE_CATEGORIES


def _make_target(name: str, directory_path: Path) -> RegisteredTarget:
    """Build a RegisteredTarget whose only test-relevant field is the path.

    Every other required field is held at a fixed valid value so registration
    outcome is attributable solely to workspace containment.
    """
    return RegisteredTarget(
        name=name,
        directory_path=directory_path,
        repo_remote="https://example.invalid/repo.git",
        credentials_ref="GIT_TOKEN_ENV",
    )


def _build_candidate(
    category: str, workspace: Path, outside: Path, segments: list[str]
) -> Path:
    """Materialize a candidate path for ``category`` and return it.

    Creates whatever real directories / symlinks the category needs inside the
    given temp ``workspace`` (in-workspace) or ``outside`` (escaping) roots.
    """
    if category == "inside_relative":
        candidate = workspace.joinpath(*segments)
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    if category == "inside_root":
        # The workspace directory itself resolves inside the workspace.
        return workspace

    if category == "inside_symlink":
        real = workspace / ("real_" + segments[0])
        real.mkdir(parents=True, exist_ok=True)
        link = workspace / ("link_" + segments[0])
        link.symlink_to(real, target_is_directory=True)
        return link

    if category == "escape_dotdot":
        # workspace/../outside/... resolves into the sibling "outside" tree,
        # which is genuinely outside the workspace.
        return workspace.joinpath("..", outside.name, *segments)

    if category == "escape_absolute":
        candidate = outside.joinpath(*segments)
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    if category == "escape_symlink":
        # A link that lives lexically inside the workspace but points outside.
        link = workspace / ("esclink_" + segments[0])
        link.symlink_to(outside, target_is_directory=True)
        return link

    raise AssertionError(f"unhandled category {category!r}")


def _registry_targets(registry_path: Path) -> list[dict]:
    """Return the persisted target entries, or ``[]`` if the file is absent."""
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return data.get("targets", []) if isinstance(data, dict) else data


# Feature: discord-ollama-coding-agent, Property 12
# Property 12: Target registration is confined to the Workspace_Directory.
# Validates: Requirements 10.3
@settings(max_examples=20)
@given(
    category=st.sampled_from(_CATEGORIES),
    segments=st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
    target_name=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=20
    ),
)
def test_registration_is_confined_to_workspace(
    category: str, segments: list[str], target_name: str
):
    """A target is registered iff its path resolves inside the workspace.

    In-workspace candidates (relative, the workspace root, and inward symlinks)
    are accepted; escaping candidates (``..`` traversal, absolute-outside, and
    outward symlinks) raise :class:`PathOutsideWorkspaceError` and leave the
    registry unchanged.
    """
    expected_inside = category in _INSIDE_CATEGORIES

    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        workspace = root / "workspace"
        workspace.mkdir()
        outside = root / "outside"
        outside.mkdir()
        registry_path = root / "config" / "registry.json"

        registry = TargetRegistry()
        # Fresh deployment: missing registry file => empty usable set.
        registry.load(registry_path, workspace)
        assert registry.names() == []

        candidate = _build_candidate(category, workspace, outside, segments)
        target = _make_target(target_name, candidate)

        if expected_inside:
            registry.add(target)
            # Accepted: present in the usable set and persisted to disk.
            assert registry.exists(target_name)
            assert registry.get(target_name) is target
            assert registry.names() == [target_name]
            persisted_names = [
                entry.get("name") for entry in _registry_targets(registry_path)
            ]
            assert persisted_names == [target_name]
        else:
            with pytest.raises(PathOutsideWorkspaceError):
                registry.add(target)
            # Rejected: registry state and backing file are untouched.
            assert not registry.exists(target_name)
            assert registry.get(target_name) is None
            assert registry.names() == []
            assert _registry_targets(registry_path) == []
