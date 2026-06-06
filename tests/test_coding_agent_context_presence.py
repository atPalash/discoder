"""Property-based test for first-attempt context presence vs. emptiness (task 9.3).

Covers Property 10: Context presence follows working-copy emptiness.

*For any* Job, the first generation request includes existing-file context **if
and only if** the Job's working copy is non-empty; an empty working copy yields
a request with no existing-file context. Here the working copy is exercised
directly through :meth:`CodingAgent._build_context`, which produces the
existing-file context for the first generation attempt: it returns ``[]`` for an
empty working copy (Req 3.2) and a non-empty selection for a non-empty one
(Req 3.1).

The oracle ("is the working copy non-empty?") is derived purely from the
generated set of readable text files (test knowledge), never by re-running the
agent's own walk, so the test independently pins down the IFF. VCS-metadata
directories such as ``.git`` are not project source, so a working copy holding
only such metadata is treated as empty.

**Validates: Requirements 3.1, 3.2**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.coding_agent import CodingAgent
from discord_ollama_agent.models.config import ContextBudget
from discord_ollama_agent.models.generation import FileEntry

# Inclusive ContextBudget bounds (mirrors models/config.ContextBudget).
_MIN_TOTAL_BYTES = 1024
_MAX_TOTAL_BYTES = 67_108_864
_MAX_FILE_COUNT = 1000

# Safe single path components: lowercase letters and digits only, so a drawn
# segment is always a legal path component and never ``.``, ``..``, or a
# VCS-metadata directory name such as ``.git``.
_SAFE_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
)

# File contents restricted to characters that encode cleanly as UTF-8 so the
# bytes written round-trip back through the agent's UTF-8 decode (a file the
# agent cannot decode would be skipped, which is a separate concern). Empty
# content is allowed: an empty but readable file is still project source.
_TEXT_CONTENT = st.text(st.characters(codec="utf-8"), max_size=64)

# A single readable text file: a relative path (1-3 segments) plus its content.
_FILE_SPEC = st.fixed_dictionaries(
    {
        "segments": st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
        "content": _TEXT_CONTENT,
    }
)


def _budget_for(total_bytes: int, file_count: int) -> ContextBudget:
    """A budget guaranteed to fit every generated file.

    Property 10 concerns presence vs. emptiness, not budget trimming (that is
    Property 9), so the budget is sized so a non-empty working copy always yields
    a non-empty selection: it admits at least as many files and bytes as were
    written (clamped to the model's valid bounds).
    """
    return ContextBudget(
        max_file_count=min(_MAX_FILE_COUNT, max(1, file_count)),
        max_total_bytes=min(_MAX_TOTAL_BYTES, max(_MIN_TOTAL_BYTES, total_bytes)),
    )


# Feature: discord-ollama-coding-agent, Property 10
# Property 10: Context presence follows working-copy emptiness
# Validates: Requirements 3.1, 3.2
@settings(max_examples=20, deadline=None)
@given(
    specs=st.lists(_FILE_SPEC, min_size=0, max_size=6),
    include_git=st.booleans(),
)
def test_context_presence_follows_working_copy_emptiness(
    specs: list[dict], include_git: bool
):
    """`_build_context` returns context iff the working copy has readable files.

    Materializes a working copy from the generated readable text files (plus
    optional ignored ``.git`` metadata), then asserts the first-attempt context
    is empty iff no project files were written, and otherwise reproduces exactly
    the files that were written (Req 3.1, 3.2).
    """
    agent = CodingAgent()

    with tempfile.TemporaryDirectory() as root_name:
        working_copy = Path(root_name)

        # Each spec lives under a unique top-level directory ("f0", "f1", ...) so
        # no two specs collide and no file path is ever a parent of another, i.e.
        # every spec yields exactly one distinct readable file.
        expected: dict[str, str] = {}
        for index, spec in enumerate(specs):
            relative = Path(f"f{index}", *spec["segments"])
            absolute = working_copy / relative
            absolute.parent.mkdir(parents=True, exist_ok=True)
            absolute.write_bytes(spec["content"].encode("utf-8"))
            expected[relative.as_posix()] = spec["content"]

        # VCS metadata is not project source: a ``.git`` entry must never make an
        # otherwise-empty working copy count as non-empty.
        if include_git:
            git_dir = working_copy / ".git"
            git_dir.mkdir(parents=True, exist_ok=True)
            (git_dir / "config").write_text("[core]\n", encoding="utf-8")

        total_bytes = sum(len(content.encode("utf-8")) for content in expected.values())
        budget = _budget_for(total_bytes, len(expected))

        result = agent._build_context(working_copy, budget)

        # The IFF: context is present exactly when the working copy held readable
        # project files. ``bool(expected)`` is the independent oracle.
        assert bool(result) == bool(expected), (
            f"expected context present={bool(expected)} for "
            f"{len(expected)} project file(s), got {len(result)} entries"
        )

        if not expected:
            # Req 3.2: an empty working copy yields no existing-file context, even
            # when it carries ignored VCS metadata.
            assert result == []
        else:
            # Req 3.1: a non-empty working copy yields its project files as
            # context. The budget fits everything, so the full set is returned.
            assert all(isinstance(entry, FileEntry) for entry in result)
            returned = {entry.path: entry.content for entry in result}
            assert returned == expected
            # Ignored VCS metadata never leaks into the context.
            assert all(not path.startswith(".git/") for path in returned)
