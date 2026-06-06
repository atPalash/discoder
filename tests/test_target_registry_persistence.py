"""Property-based test for registration persistence across reloads (task 4.4).

Covers Property 24: target registration persists across reloads.

*For any* valid target added via ``TargetRegistry.add`` (the persistence step
behind ``/addtarget``), reloading the Target_Registry from disk with a fresh
``TargetRegistry`` instance -- as happens after a Container restart -- yields a
registry whose usable set contains the added target with **identical**
configuration: the same ``name``, ``directory_path``, ``build_command``,
``repo_remote``, ``default_branch``, ``branch_scheme``, and ``credentials_ref``.

The test exercises the real atomic write + JSON reload round trip against a
temp-dir registry file and a temp workspace holding the target directory; no
state is shared between the writing registry and the reloading registry beyond
the on-disk file, so a passing assertion proves the configuration genuinely
survived serialization to and from disk.

**Validates: Requirements 10.1**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.target_registry import TargetRegistry

# Target names: at least one and at most 40 non-whitespace characters drawn from
# a name-safe alphabet, so the model's name bounds (non-whitespace, <= 64 chars)
# always hold and registration outcome turns purely on persistence.
_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=40,
)

# Directory segments: lowercase letters and digits only, so a drawn segment is
# always a legal single path component (never ``.`` or ``..``) inside the
# workspace.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
)

# Non-blank config text: printable ASCII excluding space (codepoints 33..126),
# so the value is always non-empty and non-whitespace. Used for fields that must
# survive the round trip with a meaningful value (remote, credentials_ref,
# default_branch, branch_scheme).
_NONBLANK_TEXT = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)

# Build command: optional. ``None`` (no build step) or any printable string,
# including spaces (e.g. "npm run build") and the empty string, all of which
# must round-trip unchanged.
_BUILD_COMMAND = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0,
        max_size=40,
    ),
)


# Feature: discord-ollama-coding-agent, Property 24: Target registration persists across reloads
# Validates: Requirements 10.1
@settings(max_examples=20, deadline=None)
@given(
    name=_NAME,
    segments=st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
    build_command=_BUILD_COMMAND,
    repo_remote=_NONBLANK_TEXT,
    default_branch=_NONBLANK_TEXT,
    branch_scheme=_NONBLANK_TEXT,
    credentials_ref=_NONBLANK_TEXT,
)
def test_registration_persists_across_reloads(
    name: str,
    segments: list[str],
    build_command: str | None,
    repo_remote: str,
    default_branch: str,
    branch_scheme: str,
    credentials_ref: str,
):
    """A valid added target reappears with identical config after a reload.

    Adds one valid target to a registry backed by a temp file, then loads a
    brand-new ``TargetRegistry`` from the same file + workspace (simulating a
    Container restart). The reloaded usable set must contain the target with
    every field byte-for-byte identical to what was registered.
    """
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        workspace = root / "workspace"
        workspace.mkdir()
        registry_path = root / "config" / "registry.json"

        # The target directory is a real, existing directory inside the
        # workspace so the add-time containment + existence checks pass and the
        # outcome is attributable to persistence alone.
        directory = workspace.joinpath(*segments)
        directory.mkdir(parents=True, exist_ok=True)

        target = RegisteredTarget(
            name=name,
            directory_path=directory,
            build_command=build_command,
            repo_remote=repo_remote,
            default_branch=default_branch,
            branch_scheme=branch_scheme,
            credentials_ref=credentials_ref,
        )

        # Register + persist atomically through the writing registry.
        writer = TargetRegistry()
        writer.load(registry_path, workspace)
        assert writer.names() == []
        writer.add(target)

        # Reload from disk with a fresh instance: nothing is shared but the file.
        reloaded = TargetRegistry()
        result = reloaded.load(registry_path, workspace)

        # The target survives as a usable target (Req 10.1).
        assert reloaded.exists(name)
        assert reloaded.names() == [name]
        assert [t.name for t in result.usable] == [name]
        assert result.excluded == []

        restored = reloaded.get(name)
        assert restored is not None

        # Identical configuration across every persisted field.
        assert restored.name == target.name
        assert restored.directory_path == target.directory_path
        assert restored.build_command == target.build_command
        assert restored.repo_remote == target.repo_remote
        assert restored.default_branch == target.default_branch
        assert restored.branch_scheme == target.branch_scheme
        assert restored.credentials_ref == target.credentials_ref

        # The whole model compares equal, guarding against any unchecked field.
        assert restored == target
