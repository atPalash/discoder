"""Property-based test for FIFO dispatch order (task 10.5).

Covers Property 7: Dispatch order is FIFO by submission time.

*For any* set of Queued Jobs, whenever a running slot becomes available the Job
selected to transition to Running is the Queued Job with the earliest submission
time (Req 7.4). In other words, the order in which workers actually *start* is
exactly the order of the Jobs sorted by ``submitted_at`` (earliest first).

The test drives the real :class:`JobManager` dispatcher with
``concurrency_limit=1`` so at most one Job runs at a time and the promotion order
is directly observable: a worker records the id of the Job it was handed at the
moment it starts, then completes immediately so the slot frees and the
dispatcher promotes the next Queued Job. The recorded start sequence must match
submission order.

To make "earliest submission time" unambiguous (so the assertion does not hinge
on sub-microsecond timestamp ties), each Job's ``submitted_at`` is set to a
strictly increasing value in creation order -- which is exactly the real
invariant ``create_job`` maintains (a Job submitted later has a later
``submitted_at`` and is enqueued behind earlier ones).

**Validates: Requirements 7.4**
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.execution_sandbox import CancellationToken
from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# A small pool of valid target names, reused across Jobs so that dispatch order
# can never come from the target name -- only from submission time.
_TARGET_NAMES = ["svc", "api", "data-pipeline", "x", "svc-1"]

# Anchor for the strictly-increasing submission timestamps assigned below.
_BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_target(name: str) -> RegisteredTarget:
    """Build a valid target whose only meaningful variable here is ``name``."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("workspace") / "t",
        repo_remote="git@example.com:org/repo.git",
        credentials_ref="GIT_TOKEN",
    )


# Feature: discord-ollama-coding-agent, Property 7: Dispatch order is FIFO by submission time
# Validates: Requirements 7.4
@settings(max_examples=20, deadline=None)
@given(target_names=st.lists(st.sampled_from(_TARGET_NAMES), min_size=1, max_size=12))
@pytest.mark.asyncio
async def test_dispatch_order_is_fifo_by_submission_time(
    target_names: list[str],
) -> None:
    """Workers start in submission order: earliest ``submitted_at`` first."""
    targets = {name: _make_target(name) for name in _TARGET_NAMES}

    # Records the id of each Job at the moment its worker starts, i.e. the order
    # in which the dispatcher promoted Queued Jobs to Running.
    start_order: list[str] = []

    async def worker(job: Job, _token: CancellationToken) -> None:
        # The promotion is observed here, before the slot is freed. Completing
        # immediately frees the single slot so the next Queued Job is promoted.
        start_order.append(job.id)
        job.status = JobStatus.SUCCEEDED

    # concurrency_limit=1 => exactly one running slot, so the promotion sequence
    # is fully serialized and directly observable (Req 7.4).
    manager = JobManager(concurrency_limit=1, worker=worker)

    # Enqueue all Jobs up front, assigning strictly increasing submission times
    # in creation order (the invariant create_job maintains). This makes the
    # "earliest submission time" target unambiguous.
    created: list[Job] = []
    for index, name in enumerate(target_names):
        job = manager.create_job(
            targets[name],
            idea="add a healthcheck endpoint",
            channel_id=100,
            user_id=7,
        )
        job.submitted_at = _BASE_TIME + timedelta(seconds=index)
        created.append(job)

    manager.start()
    manager.notify()

    # Wait until every Job has been promoted and run, then stop the dispatcher.
    try:
        async def _all_dispatched() -> None:
            while len(start_order) < len(created):
                await asyncio.sleep(0)

        await asyncio.wait_for(_all_dispatched(), timeout=5.0)
    finally:
        await manager.stop()

    # The expected promotion order is the Jobs sorted by submission time,
    # earliest first (Req 7.4). With strictly increasing submitted_at this is
    # exactly the submission (creation) order.
    expected_order = [job.id for job in sorted(created, key=lambda j: j.submitted_at)]

    assert start_order == expected_order, (
        "dispatch order did not follow earliest-submission-time-first: "
        f"expected {expected_order!r}, got {start_order!r}"
    )

    # The submission timestamps along the actual start order must be strictly
    # increasing, confirming each promotion picked the earliest Queued Job.
    submitted_by_id = {job.id: job.submitted_at for job in created}
    started_times = [submitted_by_id[job_id] for job_id in start_order]
    assert started_times == sorted(started_times), (
        "submission times along the dispatch order are not non-decreasing: "
        f"{started_times!r}"
    )
