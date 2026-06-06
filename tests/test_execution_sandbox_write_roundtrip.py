"""Property-based test for sandbox file-write round-trip (task 5.3).

Covers Property 13: file writes round-trip within the sandbox.

*For any* sequence of in-bounds file entries written to a Job's working copy via
:meth:`ExecutionSandbox.write_file`, reading each written path back yields
exactly the content that was written, and when the same path is written more
than once the later write overwrites the earlier content. The expected final
state is computed independently by the test (a last-writer-wins mapping built
from the drawn sequence), so the test is an independent oracle of Req 4.1 rather
than a re-run of the implementation's own logic.

Paths are drawn from a small fixed pool of mutually non-conflicting relative
paths so that (a) the same path recurs across a sequence, exercising the
overwrite (later-write-wins) behaviour, and (b) no path is ever both a regular
file and an ancestor directory of another -- a contradictory shape that a real
"set of file entries" never has, and which is outside the round-trip property's
input space. Directory segments (``dirN``/``sub``) and file leaves (``*.txt``)
are drawn from disjoint name shapes so a leaf can never double as a directory.

Reading back uses the raw persisted bytes decoded as UTF-8 (rather than a
text-mode read) so the comparison reflects exactly what was written to disk,
without text-mode universal-newline translation rewriting characters such as
``\\r``.

**Validates: Requirements 4.1**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.execution_sandbox import ExecutionSandbox
from discord_ollama_agent.models.job import Job

# A fixed pool of mutually non-conflicting in-bounds relative paths. No entry is
# a prefix (ancestor directory) of another, and the ``dirN``/``sub`` segments are
# only ever directories while ``*.txt`` segments are only ever file leaves, so
# writing any subset in any order never makes a path serve as both a file and a
# directory. The pool is small so paths recur within a sequence, exercising
# overwrite behaviour, while still covering top-level and nested files.
_PATH_POOL = (
    "a.txt",
    "b.txt",
    "dir1/c.txt",
    "dir1/d.txt",
    "dir2/sub/e.txt",
)

# A relative path drawn from the non-conflicting pool, normalized for the
# current platform.
_RELATIVE_PATH = st.sampled_from(_PATH_POOL).map(lambda p: str(Path(p)))

# File contents, including the empty string, unicode, and newline characters,
# since write_file does a whole-file UTF-8 write.
_CONTENT = st.text(max_size=128)

# A sequence of (relative_path, content) writes applied in order. min_size=1 so
# at least one write always happens.
_WRITE_ENTRIES = st.lists(
    st.tuples(_RELATIVE_PATH, _CONTENT),
    min_size=1,
    max_size=20,
)


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


def _read_back(path: Path) -> str:
    """Read the persisted file as raw UTF-8, without newline translation.

    Comparing against the raw bytes reflects exactly what ``write_file`` put on
    disk; a text-mode read would translate newline characters and mask a true
    round-trip.
    """
    return path.read_bytes().decode("utf-8")


# Feature: discord-ollama-coding-agent, Property 13
# Property 13: File writes round-trip within the sandbox.
# Validates: Requirements 4.1
@settings(max_examples=20)
@given(entries=_WRITE_ENTRIES)
def test_in_bounds_writes_round_trip_with_last_writer_wins(
    entries: list[tuple[str, str]],
):
    """Reading each written path back yields exactly the last content written.

    Each write is immediately readable as exactly its content (round-trip), and
    after the full sequence every path holds the content of its final write
    (later writes overwrite earlier ones).
    """
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        sandbox_root = root / "sandbox_root"
        sandbox_root.mkdir()

        sandbox = ExecutionSandbox(sandbox_root)
        working_dir = sandbox.init_working_dir(_make_job(), source=root / "missing")

        # Independent oracle: last-writer-wins mapping of resolved path -> content.
        expected: dict[Path, str] = {}

        for relative_path, content in entries:
            sandbox.write_file(relative_path, content)
            resolved = sandbox.resolve_within(relative_path)

            # The just-written file round-trips exactly and stays in bounds.
            assert resolved.is_relative_to(working_dir.resolve())
            assert _read_back(resolved) == content

            expected[resolved] = content

        # After the whole sequence, every path holds its final write's content,
        # i.e. later writes overwrote earlier writes to the same path.
        for resolved, content in expected.items():
            assert _read_back(resolved) == content
