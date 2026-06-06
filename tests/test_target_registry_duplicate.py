"""Property-based test for duplicate-name registration rejection (task 4.5).

Covers Property 25: duplicate target names are rejected without clobbering.

*For any* candidate target whose name matches a Registered_Target already in the
Target_Registry, ``TargetRegistry.add`` rejects the registration with
:class:`DuplicateTargetError` and the existing target's configuration is left
entirely unchanged -- in the usable set, and in the persisted backing file --
even when the rejected candidate carries an otherwise-valid (but different)
directory, remote, branch scheme, and credentials reference. This is the
guarantee behind Req 10.5: a name collision SHALL NOT overwrite the existing
target.

The test backs the registry with a temp file and a temp workspace holding two
distinct, real directories so the only reason the second ``add`` can fail is the
duplicate name (its path and existence checks would otherwise pass). A passing
assertion therefore proves the duplicate guard fires *and* protects the
incumbent's configuration from being clobbered.

**Validates: Requirements 10.5**
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.errors import DuplicateTargetError
from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.target_registry import TargetRegistry

# Target names: at least one and at most 40 non-whitespace characters from a
# name-safe alphabet, so the model's name bounds always hold and the only
# registration failure on the second add is the duplicate name.
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

# Non-blank config text: printable ASCII excluding space, so the value is always
# non-empty and non-whitespace (remote, credentials_ref, branch fields).
_NONBLANK_TEXT = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)

_BUILD_COMMAND = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0,
        max_size=40,
    ),
)

# A full per-target configuration bundle (everything but the shared name and the
# directory, which are supplied separately so the two targets genuinely differ).
_CONFIG = st.fixed_dictionaries(
    {
        "build_command": _BUILD_COMMAND,
        "repo_remote": _NONBLANK_TEXT,
        "default_branch": _NONBLANK_TEXT,
        "branch_scheme": _NONBLANK_TEXT,
        "credentials_ref": _NONBLANK_TEXT,
    }
)


def _persisted_targets(registry_path: Path) -> list[dict]:
    """Return the persisted target entries, or ``[]`` if the file is absent."""
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return data.get("targets", []) if isinstance(data, dict) else data


# Feature: discord-ollama-coding-agent, Property 25: Duplicate target names are rejected without clobbering
# Validates: Requirements 10.5
@settings(max_examples=20, deadline=None)
@given(
    name=_NAME,
    existing_segments=st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
    existing_config=_CONFIG,
    intruder_segments=st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
    intruder_config=_CONFIG,
)
def test_duplicate_name_is_rejected_without_clobbering(
    name: str,
    existing_segments: list[str],
    existing_config: dict,
    intruder_segments: list[str],
    intruder_config: dict,
):
    """A name collision is rejected and the incumbent target is untouched.

    Registers an initial valid target, then attempts to register a second,
    otherwise-valid target that shares the same name but differs in every other
    field. The second ``add`` must raise :class:`DuplicateTargetError`, and the
    existing target -- both in the usable set and on disk -- must retain its
    original configuration unchanged.
    """
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        workspace = root / "workspace"
        workspace.mkdir()
        registry_path = root / "config" / "registry.json"

        # Two distinct, real directories inside the workspace so both candidates
        # would pass the containment + existence checks; only the duplicate name
        # can cause the second add to fail. "a_"/"b_" prefixes guarantee the two
        # directories never coincide even if the drawn segments match.
        existing_dir = workspace.joinpath("a_" + existing_segments[0], *existing_segments[1:])
        existing_dir.mkdir(parents=True, exist_ok=True)
        intruder_dir = workspace.joinpath("b_" + intruder_segments[0], *intruder_segments[1:])
        intruder_dir.mkdir(parents=True, exist_ok=True)

        existing = RegisteredTarget(
            name=name,
            directory_path=existing_dir,
            build_command=existing_config["build_command"],
            repo_remote=existing_config["repo_remote"],
            default_branch=existing_config["default_branch"],
            branch_scheme=existing_config["branch_scheme"],
            credentials_ref=existing_config["credentials_ref"],
        )
        intruder = RegisteredTarget(
            name=name,
            directory_path=intruder_dir,
            build_command=intruder_config["build_command"],
            repo_remote=intruder_config["repo_remote"],
            default_branch=intruder_config["default_branch"],
            branch_scheme=intruder_config["branch_scheme"],
            credentials_ref=intruder_config["credentials_ref"],
        )

        registry = TargetRegistry()
        registry.load(registry_path, workspace)
        registry.add(existing)

        # Snapshot the persisted state before the rejected attempt.
        before_entries = _persisted_targets(registry_path)
        assert [e.get("name") for e in before_entries] == [name]

        # (1) Registering a same-named target is rejected.
        with pytest.raises(DuplicateTargetError):
            registry.add(intruder)

        # (2) The existing target's configuration is unchanged in the usable set.
        kept = registry.get(name)
        assert kept is not None
        assert kept == existing
        assert kept.directory_path == existing_dir
        assert kept.repo_remote == existing_config["repo_remote"]
        assert kept.build_command == existing_config["build_command"]
        assert kept.default_branch == existing_config["default_branch"]
        assert kept.branch_scheme == existing_config["branch_scheme"]
        assert kept.credentials_ref == existing_config["credentials_ref"]
        assert registry.names() == [name]

        # ...and unchanged on disk: the backing file still holds exactly the
        # original entry, never the intruder's clobbering configuration.
        assert _persisted_targets(registry_path) == before_entries

        # A fresh reload from disk still yields the original target verbatim,
        # proving nothing of the intruder's config leaked into the registry.
        reloaded = TargetRegistry()
        reloaded.load(registry_path, workspace)
        restored = reloaded.get(name)
        assert restored is not None
        assert restored == existing
