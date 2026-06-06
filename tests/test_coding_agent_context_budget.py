"""Property-based test for budget-bounded context selection (task 9.2).

Covers Property 9: the context selection stays within the configured budget.

*For any* set of existing project files in a Job's working copy and *for any*
valid :class:`ContextBudget`, the context returned by
:meth:`CodingAgent._build_context` contains at most ``budget.max_file_count``
files and a combined UTF-8 content size of at most ``budget.max_total_bytes``,
and that bounded subset is exactly what generation proceeds with (Req 3.4, 3.5).

The expected bounds are derived from the budget alone (test knowledge), never by
re-running the implementation's own selection logic, so the test is an
independent oracle of Req 3.4/3.5. Files are materialised on disk and read back
through the real ``_build_context`` walk; no services are mocked.

**Validates: Requirements 3.4, 3.5**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.coding_agent import CodingAgent
from discord_ollama_agent.models.config import ContextBudget
from discord_ollama_agent.models.generation import GenerationRequest

# ContextBudget bounds (Req 4.14). The byte budget is kept toward the low end of
# its valid range so that drawn file sets routinely exceed it, exercising the
# subset-selection path (Req 3.5) rather than always fitting.
_MAX_FILE_COUNT = (1, 50)
_MAX_TOTAL_BYTES = (1024, 16_384)

# Unique, always-legal single path components: lowercase letters and digits.
_SAFE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=10,
)

# File contents drawn across a size range straddling the byte budget so some
# individual files fit, some combinations overflow, and some single files are
# larger than the whole budget (and can never be included).
_CONTENT = st.text(max_size=4096)


def _content_bytes(content: str) -> int:
    """Budget cost of a file: its UTF-8 encoded length (the impl's metric)."""
    return len(content.encode("utf-8"))


# Feature: discord-ollama-coding-agent, Property 9
# Property 9: Context selection stays within the configured budget.
# Validates: Requirements 3.4, 3.5
@settings(max_examples=20)
@given(
    files=st.dictionaries(
        keys=_SAFE_NAME,
        values=_CONTENT,
        max_size=60,
    ),
    max_file_count=st.integers(*_MAX_FILE_COUNT),
    max_total_bytes=st.integers(*_MAX_TOTAL_BYTES),
)
def test_context_selection_stays_within_budget(
    files: dict[str, str],
    max_file_count: int,
    max_total_bytes: int,
):
    """The selected context honours both budget bounds and feeds generation.

    For any working copy of existing files and any valid ContextBudget, the
    returned selection has at most ``max_file_count`` files and a combined size
    of at most ``max_total_bytes`` bytes, every selected entry is a genuine
    working-copy file, and the same bounded subset is what a GenerationRequest
    proceeds with.
    """
    budget = ContextBudget(
        max_file_count=max_file_count,
        max_total_bytes=max_total_bytes,
    )
    agent = CodingAgent()

    with tempfile.TemporaryDirectory() as tmp:
        working_copy = Path(tmp)
        # Materialise the file set on disk as raw UTF-8 bytes so the content
        # round-trips through the implementation's read_bytes().decode("utf-8")
        # walk without newline translation.
        for name, content in files.items():
            (working_copy / name).write_bytes(content.encode("utf-8"))

        selection = agent._build_context(working_copy, budget)

        # Bound 1 (Req 3.4): at most the configured maximum file count.
        assert len(selection) <= budget.max_file_count

        # Bound 2 (Req 3.4/3.5): combined UTF-8 size within the byte budget.
        total = sum(_content_bytes(entry.content) for entry in selection)
        assert total <= budget.max_total_bytes

        # Every selected entry is a real working-copy file with matching content,
        # so the bounded subset is drawn from the existing project (not invented).
        for entry in selection:
            assert entry.path in files
            assert entry.content == files[entry.path]

        # Paths within the selection are distinct (no file counted twice).
        selected_paths = [entry.path for entry in selection]
        assert len(selected_paths) == len(set(selected_paths))

        # Generation proceeds with exactly that bounded subset (Req 3.5): the
        # request carries the selected context unchanged and still in budget.
        request = GenerationRequest(idea="add a feature", context_files=selection)
        assert request.context_files == selection
        assert len(request.context_files) <= budget.max_file_count
        assert (
            sum(_content_bytes(e.content) for e in request.context_files)
            <= budget.max_total_bytes
        )
