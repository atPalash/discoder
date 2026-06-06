"""Property-based test for sandbox write confinement (task 5.2).

Covers Property 11: the Execution_Sandbox confines all writes to the Job's
working directory.

*For any* relative path handed to :meth:`ExecutionSandbox.write_file`, the write
is permitted **if and only if** the path resolves to a location inside the Job's
working directory. In-bounds relative paths (including nested directories and
paths reached through a symlink that resolves back inside) are written
successfully. Escaping paths -- absolute paths, parent-directory (``..``)
traversal, and symlinks that resolve outside -- are denied with
:class:`PathEscapeError`, and no file outside the working directory is modified.

The expected outcome is derived from the path *category* (test knowledge), never
by re-running the implementation's own resolve logic, so the test is an
independent oracle of Req 4.2.

**Validates: Requirements 4.2**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.errors import PathEscapeError
from discord_ollama_agent.execution_sandbox import ExecutionSandbox
from discord_ollama_agent.models.job import Job

# Safe path segments: lowercase letters and digits only, so a drawn segment is
# always a legal single path component and never ``.`` or ``..``.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=8,
)

# ``inside_*`` categories genuinely resolve inside the working directory;
# ``escape_*`` categories genuinely resolve outside it. The expected-permitted
# oracle is derived from the category alone.
_INSIDE_CATEGORIES = ("inside_relative", "inside_nested", "inside_symlink")
_ESCAPE_CATEGORIES = ("escape_dotdot", "escape_absolute", "escape_symlink")
_CATEGORIES = _INSIDE_CATEGORIES + _ESCAPE_CATEGORIES

# A canary file planted outside the working directory; a denied write must never
# alter it.
_CANARY_CONTENT = "canary-untouched"


def _make_job() -> Job:
    """Build a minimal valid Job to seed the sandbox working directory."""
    return Job(
        id="example-target-0001",
        target_name="example-target",
        idea="add a feature",
        submitted_at=datetime.now(timezone.utc),
        channel_id=1,
        user_id=2,
    )


def _build_relative_path(
    category: str, working_dir: Path, outside: Path, segments: list[str]
) -> str:
    """Return the relative-path argument to ``write_file`` for ``category``.

    For categories that depend on a pre-existing symlink, the symlink is created
    inside ``working_dir`` first so the resolution behaviour is realistic.
    """
    if category == "inside_relative":
        return segments[0]

    if category == "inside_nested":
        return str(Path(*segments))

    if category == "inside_symlink":
        # A directory symlink that lives inside the working dir and points to a
        # real in-bounds directory; writing through it stays in bounds.
        real = working_dir / ("real_" + segments[0])
        real.mkdir(parents=True, exist_ok=True)
        link = working_dir / ("link_" + segments[0])
        link.symlink_to(real, target_is_directory=True)
        return str(Path(link.name, segments[-1]))

    if category == "escape_dotdot":
        # ``../<outside>/...`` climbs out of the working dir into a sibling tree.
        return str(Path("..", outside.name, *segments))

    if category == "escape_absolute":
        # An absolute path outside the working dir; joining resets to it.
        return str(outside.joinpath(*segments))

    if category == "escape_symlink":
        # A directory symlink inside the working dir that points outside it;
        # writing through it would escape.
        link = working_dir / ("esclink_" + segments[0])
        link.symlink_to(outside, target_is_directory=True)
        return str(Path(link.name, segments[-1]))

    raise AssertionError(f"unhandled category {category!r}")


def _files_under(root: Path) -> dict[Path, str]:
    """Snapshot every regular file under ``root`` as a path->content mapping."""
    return {
        p: p.read_text(encoding="utf-8")
        for p in root.rglob("*")
        if p.is_file()
    }


# Feature: discord-ollama-coding-agent, Property 11
# Property 11: Sandbox confines all writes to the Job working directory.
# Validates: Requirements 4.2
@settings(max_examples=20)
@given(
    category=st.sampled_from(_CATEGORIES),
    segments=st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
    # Exclude carriage returns: text-mode read applies universal-newline
    # translation, which is Property 13's (round-trip) concern, not this
    # confinement property's.
    content=st.text(alphabet=st.characters(exclude_characters="\r"), max_size=64),
)
def test_writes_are_confined_to_working_dir(
    category: str, segments: list[str], content: str
):
    """A write is permitted iff its path resolves inside the working dir.

    In-bounds paths are written and round-trip; escaping paths raise
    :class:`PathEscapeError` and leave every file outside the working directory
    untouched.
    """
    expected_inside = category in _INSIDE_CATEGORIES

    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        # Sibling trees: the sandbox root (working dirs live beneath it) and an
        # outside tree holding a canary file escaping writes must not touch.
        sandbox_root = root / "sandbox_root"
        sandbox_root.mkdir()
        outside = root / "outside"
        outside.mkdir()
        canary = outside / "canary.txt"
        canary.write_text(_CANARY_CONTENT, encoding="utf-8")

        sandbox = ExecutionSandbox(sandbox_root)
        working_dir = sandbox.init_working_dir(_make_job(), source=root / "missing")

        relative_path = _build_relative_path(
            category, working_dir, outside, segments
        )

        if expected_inside:
            sandbox.write_file(relative_path, content)
            # Permitted: the resolved target exists, is in bounds, and round-trips.
            resolved = sandbox.resolve_within(relative_path)
            assert resolved.is_relative_to(working_dir.resolve())
            assert resolved.read_text(encoding="utf-8") == content
        else:
            outside_before = _files_under(outside)
            with pytest.raises(PathEscapeError):
                sandbox.write_file(relative_path, content)
            # Denied: nothing outside the working dir changed, canary intact.
            assert _files_under(outside) == outside_before
            assert canary.read_text(encoding="utf-8") == _CANARY_CONTENT
