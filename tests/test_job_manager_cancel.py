"""Property-based test for cancellation outcome (task 10.6).

Covers Property 30: Cancellation outcome depends only on current status.

*For any* Job, a :meth:`JobManager.cancel` invocation's outcome is a pure
function of the Job's status at the moment of the call:

- **Queued** -> the Job is removed from the FIFO queue and set Cancelled, and the
  outcome is ``CANCELLED_QUEUED`` (Req 8.3).
- **Running** -> the Job's :class:`CancellationToken` is signalled (which
  terminates any active build/test command) and the Job is set Cancelled, and
  the outcome is ``CANCELLED_RUNNING`` (Req 8.4).
- **Succeeded / Failed / Cancelled (terminal)** -> the request is rejected, the
  status is left unchanged, and the outcome is ``REJECTED_TERMINAL`` (Req 8.6).
- **Unknown id** -> the outcome is ``NOT_FOUND`` and nothing is mutated.

The Job is exercised through the real :meth:`JobManager.cancel` against a
manager whose state is set up to match each status: a freshly created Job for
Queued, a Job promoted to Running with a live cancellation token registered
(mirroring :meth:`JobManager._spawn_worker`), and a Job driven to each terminal
state. Generating the status across all categories is exactly what makes the
"outcome depends only on status" claim observable.

**Validates: Requirements 8.3, 8.6**
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.execution_sandbox import CancellationToken
from discord_ollama_agent.job_manager import CancelResult, JobManager
from discord_ollama_agent.models.job import JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# Status categories a Job (or a missing id) may be in when ``/cancel`` is
# invoked. These cover every branch of the cancellation outcome.
_QUEUED = "queued"
_RUNNING = "running"
_SUCCEEDED = "succeeded"
_FAILED = "failed"
_CANCELLED = "cancelled"
_UNKNOWN = "unknown"

_TERMINAL_CATEGORIES = {
    _SUCCEEDED: JobStatus.SUCCEEDED,
    _FAILED: JobStatus.FAILED,
    _CANCELLED: JobStatus.CANCELLED,
}

# A small pool of valid target names; the target name is incidental here, so a
# simple, always-valid set keeps the focus on the status-driven outcome.
_TARGET_NAMES = ["svc", "api", "data-pipeline", "x"]


def _make_target(name: str) -> RegisteredTarget:
    """Build a valid target whose only meaningful variable here is ``name``."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("workspace") / "t",
        repo_remote="git@example.com:org/repo.git",
        credentials_ref="GIT_TOKEN",
    )


# Feature: discord-ollama-coding-agent, Property 30
# Property 30: Cancellation outcome depends only on current status.
# Validates: Requirements 8.3, 8.6
@settings(max_examples=20, deadline=None)
@given(
    category=st.sampled_from(
        [_QUEUED, _RUNNING, _SUCCEEDED, _FAILED, _CANCELLED, _UNKNOWN]
    ),
    target_name=st.sampled_from(_TARGET_NAMES),
    idea=st.text(min_size=1, max_size=60),
    channel_id=st.integers(min_value=1, max_value=1_000_000),
    user_id=st.integers(min_value=1, max_value=1_000_000),
)
@pytest.mark.asyncio
async def test_cancellation_outcome_depends_only_on_current_status(
    category: str,
    target_name: str,
    idea: str,
    channel_id: int,
    user_id: int,
):
    """The cancel outcome is a pure function of the Job's current status."""
    manager = JobManager()
    target = _make_target(target_name)

    if category == _UNKNOWN:
        # No Job is created with this id, so the lookup must report not-found
        # and nothing is mutated (Req 8.2).
        outcome = await manager.cancel("no-such-job-id")
        assert outcome.result is CancelResult.NOT_FOUND
        assert outcome.job is None
        assert outcome.not_found
        assert not outcome.cancelled
        return

    # Every remaining category starts from a freshly created (Queued, queued)
    # Job; we then move it into the state under test.
    job = manager.create_job(
        target, idea=idea, channel_id=channel_id, user_id=user_id
    )
    assert job.id in manager._queue

    if category == _QUEUED:
        # A Queued Job is removed from the queue and set Cancelled (Req 8.3).
        outcome = await manager.cancel(job.id)

        assert outcome.result is CancelResult.CANCELLED_QUEUED
        assert outcome.cancelled
        assert outcome.job is job
        assert job.status is JobStatus.CANCELLED
        # Removed from the FIFO queue so the dispatcher never promotes it.
        assert job.id not in manager._queue
        return

    if category == _RUNNING:
        # Mirror JobManager._spawn_worker: a Running Job has been popped from the
        # queue, has a live cancellation token, and is in status Running.
        token = CancellationToken()
        manager._cancel_tokens[job.id] = token
        manager._remove_from_queue(job.id)
        job.status = JobStatus.RUNNING
        assert not token.cancelled

        outcome = await manager.cancel(job.id)

        # Signalled to stop (token cancelled, terminating its build/test
        # command) and set Cancelled (Req 8.4).
        assert outcome.result is CancelResult.CANCELLED_RUNNING
        assert outcome.cancelled
        assert outcome.job is job
        assert job.status is JobStatus.CANCELLED
        assert token.cancelled
        return

    # Terminal categories: Succeeded, Failed, or already Cancelled. Cancellation
    # is rejected and the status is left unchanged (Req 8.6).
    terminal_status = _TERMINAL_CATEGORIES[category]
    manager._remove_from_queue(job.id)
    job.status = terminal_status

    outcome = await manager.cancel(job.id)

    assert outcome.result is CancelResult.REJECTED_TERMINAL
    assert outcome.rejected_terminal
    assert not outcome.cancelled
    assert outcome.job is job
    # Status is unchanged by the rejected cancellation.
    assert job.status is terminal_status
