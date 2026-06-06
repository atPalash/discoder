"""Property-based test for build-output capping (task 5.4).

Covers Property 15: captured build output is capped at the configured maximum
(1 MiB in production, 1,048,576 bytes).

*For any* build command output of any size, the output captured by
:meth:`ExecutionSandbox.run_command` never exceeds the sandbox's configured
``build_output_cap_bytes``. The cap is the only bound that matters here, so the
test drives the real async subprocess with outputs of varying sizes -- below,
at, and above the cap -- and asserts the retained output is always within the
cap, while remaining a faithful prefix of what the command actually produced.

A deliberately small cap is used so each example runs fast; the capping logic is
size-independent, so a small cap exercises the same code path as the 1 MiB
default.

**Validates: Requirements 4.6**
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import example, given, settings
from hypothesis import strategies as st

from discord_ollama_agent.execution_sandbox import ExecutionSandbox
from discord_ollama_agent.models.job import Job

# A small cap keeps every example trivially fast while exercising the exact same
# capping branch as the 1 MiB production default.
_CAP_BYTES = 256

# Generous timeout: each command only `cat`s at most a few KiB, so it always
# finishes well within this bound (no timeout/cancellation is under test here).
_TIMEOUT_S = 30


def _make_job() -> Job:
    """Build a minimal Job suitable for seeding a sandbox working directory."""
    return Job(
        id="example-target-0001",
        target_name="example-target",
        idea="emit output of a generated size",
        submitted_at=datetime.now(timezone.utc),
        channel_id=1,
        user_id=1,
    )


# Feature: discord-ollama-coding-agent, Property 15
# Validates: Requirements 4.6
@settings(max_examples=20, deadline=None)
@given(output_size=st.integers(min_value=0, max_value=4 * _CAP_BYTES))
@example(output_size=0)
@example(output_size=_CAP_BYTES - 1)
@example(output_size=_CAP_BYTES)
@example(output_size=_CAP_BYTES + 1)
def test_run_command_output_never_exceeds_cap(output_size: int) -> None:
    """Captured output is capped at ``build_output_cap_bytes`` for any size.

    Writes a file of exactly ``output_size`` single-byte characters into the
    Job's working copy, runs ``cat`` on it (which emits precisely that many
    bytes, no trailing newline), and asserts:

    * the captured output is at most the configured cap (Req 4.6);
    * the captured output is the leading prefix of the full produced output, so
      nothing beyond truncation is lost or fabricated;
    * the ``output_truncated`` flag is set exactly when the produced output
      exceeded the cap.
    """
    # Pure-ASCII content: 1 char == 1 byte in UTF-8, so character length and
    # byte length coincide, making the byte-cap assertion exact.
    content = "a" * output_size

    with tempfile.TemporaryDirectory() as root_name, tempfile.TemporaryDirectory() as source_name:
        sandbox = ExecutionSandbox(Path(root_name), build_output_cap_bytes=_CAP_BYTES)
        sandbox.init_working_dir(_make_job(), Path(source_name))
        sandbox.write_file("payload.txt", content)

        result = asyncio.run(
            sandbox.run_command("cat payload.txt", cancel=None, timeout_s=_TIMEOUT_S)
        )

    captured_bytes = result.output.encode("utf-8")

    # Core Property 15 invariant: never more than the configured cap.
    assert len(captured_bytes) <= _CAP_BYTES

    # The retained output is a genuine prefix of the full produced output (so the
    # cap truncates rather than corrupting), and the truncation flag is accurate.
    expected_retained = min(output_size, _CAP_BYTES)
    assert len(captured_bytes) == expected_retained
    assert result.output == content[:expected_retained]
    assert result.output_truncated == (output_size > _CAP_BYTES)
