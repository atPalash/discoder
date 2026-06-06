"""Property-based test for registry validation partition + unregistered rejection (task 4.3).

Covers Property 23: registry validation partitions targets and unregistered
builds are rejected.

*For any* loaded Target_Registry containing a mix of valid and invalid targets,
a Registered_Target is placed in the **usable** set **if and only if** its
required per-target configuration is present -- namely a ``directory_path`` that
resolves inside the Workspace_Directory **and** a non-empty Target_Repository
``repo_remote``. Targets missing either requirement are excluded, recorded in
the operator log with their name and the missing/offending value, and a lookup
(:meth:`TargetRegistry.get`) for any name not in the usable set -- whether it was
excluded by validation or never registered at all -- is rejected as
not-registered (returns ``None``).

The usable/excluded oracle is derived purely from the generated category of each
target (test knowledge), never by re-running the registry's own resolve/validate
logic, so the test independently pins down the IFF.

**Validates: Requirements 1.6, 12.2, 12.3, 12.4**
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.target_registry import TargetRegistry

_REGISTRY_LOGGER = "discord_ollama_agent.target_registry"

# Safe path/name fragments: lowercase letters and digits only, so a drawn
# fragment is always a legal single path component and never ``.`` or ``..``.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
)
_NAME_SUFFIX = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=0, max_size=10
)

# A single generated target spec. ``path_inside`` and ``remote_present`` are the
# two required-configuration dimensions; a target is usable iff BOTH hold.
_TARGET_SPEC = st.fixed_dictionaries(
    {
        "path_inside": st.booleans(),
        "remote_present": st.booleans(),
        "suffix": _NAME_SUFFIX,
        "segments": st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
        # Used only when ``remote_present`` is False: a "present but missing"
        # remote value (empty or whitespace-only) the registry must reject.
        "blank_remote": st.sampled_from(["", " ", "\t", "   ", "\n"]),
    }
)

_PRESENT_REMOTE = "https://example.invalid/repo.git"


class _CollectingHandler(logging.Handler):
    """A logging handler that retains the ERROR records emitted during load."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# Feature: discord-ollama-coding-agent, Property 23: Registry validation partitions targets and unregistered builds are rejected
# Validates: Requirements 1.6, 12.2, 12.3, 12.4
@settings(max_examples=20, deadline=None)
@given(specs=st.lists(_TARGET_SPEC, min_size=0, max_size=8))
def test_registry_partitions_targets_and_rejects_unregistered(specs: list[dict]):
    """Usable iff in-workspace path AND non-empty remote; others not-registered.

    Builds a registry file from the generated specs, loads it, and asserts:

    * the usable set is exactly the targets with both an in-workspace path and a
      non-empty remote (Req 12.3), in file order;
    * every other target is excluded with a logged reason naming the missing
      value (Req 12.2);
    * ``get`` resolves usable targets but reports excluded and never-registered
      names as not-registered (Req 1.6, 12.4).
    """
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        workspace = root / "workspace"
        workspace.mkdir()
        outside = root / "outside"
        outside.mkdir()
        registry_path = root / "config" / "registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the file-shaped entries plus an independent expectation record.
        # Names are made unique by an index prefix so no two entries collide
        # (duplicate-name exclusion is a separate property and is kept out here).
        entries: list[dict] = []
        expected: list[dict] = []
        for index, spec in enumerate(specs):
            name = f"t{index}_{spec['suffix']}"
            if spec["path_inside"]:
                directory = workspace.joinpath(*spec["segments"])
            else:
                directory = outside.joinpath(*spec["segments"])
            remote = _PRESENT_REMOTE if spec["remote_present"] else spec["blank_remote"]

            entries.append(
                {
                    "name": name,
                    "directory_path": str(directory),
                    "repo_remote": remote,
                    "credentials_ref": "GIT_TOKEN_ENV",
                }
            )
            expected.append(
                {
                    "name": name,
                    "path_inside": spec["path_inside"],
                    "remote_present": spec["remote_present"],
                    "usable": spec["path_inside"] and spec["remote_present"],
                }
            )

        registry_path.write_text(
            json.dumps({"targets": entries}, ensure_ascii=False), encoding="utf-8"
        )

        # Capture the operator-log error records emitted while partitioning.
        logger = logging.getLogger(_REGISTRY_LOGGER)
        handler = _CollectingHandler()
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        registry = TargetRegistry()
        try:
            result = registry.load(registry_path, workspace)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

        expected_usable_names = [e["name"] for e in expected if e["usable"]]
        expected_excluded = [e for e in expected if not e["usable"]]

        # Req 12.3: the usable partition is exactly the IFF oracle, in file order.
        assert registry.names() == expected_usable_names
        assert [t.name for t in result.usable] == expected_usable_names

        # Req 12.2: every non-usable target is excluded (and nothing else is).
        assert {x.name for x in result.excluded} == {e["name"] for e in expected_excluded}

        logged_messages = [record.getMessage() for record in handler.records]

        for e in expected:
            if e["usable"]:
                # Req 12.3: usable targets resolve with their stored config.
                target = registry.get(e["name"])
                assert target is not None
                assert target.name == e["name"]
                assert target.repo_remote.strip() != ""
                assert registry.exists(e["name"])
            else:
                # Req 1.6 + 12.4: a non-usable name is rejected as not-registered,
                # yet remains a known name so the duplicate guard still spans it.
                assert registry.get(e["name"]) is None
                assert registry.exists(e["name"])

                # Req 12.2: excluded with a non-empty reason naming the bad value,
                # and logged to the operator log with the target's name.
                reason = next(x.reason for x in result.excluded if x.name == e["name"])
                assert reason.strip() != ""
                quoted_name = repr(e["name"])
                matching = [m for m in logged_messages if quoted_name in m]
                assert matching, f"excluded target {e['name']!r} was not logged"
                message = matching[0].lower()
                # The remote is validated before the path, so the reason names
                # whichever requirement is unmet first.
                if not e["remote_present"]:
                    assert "remote" in message
                else:  # remote present, so the path must be the offending value
                    assert "workspace" in message or "outside" in message

        # A name that was never in the registry at all is also not-registered.
        absent = "absent_name_never_registered"
        assert registry.get(absent) is None
        assert registry.exists(absent) is False
