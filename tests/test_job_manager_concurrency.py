"""Property-based test for the concurrency invariant (task 10.4).

Covers Property 6: Running Jobs never exceed the concurrency limit.

*For any* interleaving of Job submissions and completions, the number of Jobs in
status Running never exceeds the configured concurrency limit, and every terminal
transition frees exactly one slot.

The test drives a real :class:`JobManager` (its dispatcher loop, semaphore, FIFO
queue, and ``_run_worker`` slot accounting) on a dedicated asyncio event loop.
Submissions and completions are interleaved by a Hypothesis
:class:`RuleBasedStateMachine`:

  - ``submit`` mints a new Job, ``notify()``s the dispatcher, and lets the loop
    settle so the dispatcher promotes Queued Jobs up to the limit (Req 7.3).
  - ``complete`` drives one currently-Running Job to a terminal state and
    releases its worker, so ``_run_worker`` frees the held slot and the
    dispatcher may promote exactly one waiting Job (Req 7.5, 7.7).

The worker is a controllable fake: each Job's worker simply awaits a per-Job
:class:`asyncio.Event`, so a Job stays Running until the test releases it. This
lets submissions and completions be interleaved arbitrarily.

The core invariant asserted after every step is

    running_count == min(concurrency_limit, number_of_non_terminal_jobs)

which captures both halves of Property 6 at once: the left bound by
``concurrency_limit`` means Running never exceeds the limit (Req 7.3, 7.7), and
the dependence on the count of non-terminal (Queued+Running) Jobs means a
terminal transition (which decrements that count by exactly one) frees exactly
one slot -- either dropping Running by one or letting exactly one Queued Job take
the freed slot (Req 7.5).

**Validates: Requirements 7.3, 7.5, 7.7**
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from discord_ollama_agent.execution_sandbox import CancellationToken
from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.job import Job, JobStatus

# Terminal states a completing Job may settle into; the slot-freeing behaviour
# is identical for all of them (Req 7.5).
_TERMINAL_STATUSES = [
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
]

# How many event-loop turns to spin after each mutation so the dispatcher can
# finish all promotions it is going to make. Each promotion needs only a handful
# of turns and at most one promotion happens per step, so this is comfortably
# more than enough for the loop to reach a stable state.
_PUMP_TURNS = 50


def _make_job(job_id: str) -> Job:
    """Build a minimal valid Queued Job with the given id."""
    return Job(
        id=job_id,
        target_name="svc",
        idea="add a healthcheck endpoint",
        status=JobStatus.QUEUED,
        submitted_at=datetime.now(timezone.utc),
        channel_id=100,
        user_id=7,
    )


class _ConcurrencyMachine(RuleBasedStateMachine):
    """Interleaves Job submissions and completions against a live JobManager.

    Subclasses set :attr:`concurrency_limit`. The machine owns its own event loop
    and pumps it after every mutation so the dispatcher's promotions complete
    before the invariant is checked.
    """

    concurrency_limit: int = 1

    def __init__(self) -> None:
        super().__init__()
        self._loop = asyncio.new_event_loop()
        # Per-Job release gate: a Job's worker blocks on its Event until the test
        # decides to complete that Job.
        self._release: dict[str, asyncio.Event] = {}
        # Ids submitted but not yet driven terminal (i.e. Queued or Running).
        self._active: set[str] = set()
        self._counter = 0

        self.manager = JobManager(
            concurrency_limit=self.concurrency_limit,
            worker=self._worker,
        )
        self._loop.run_until_complete(self._start())

    async def _start(self) -> None:
        # start() uses asyncio.create_task, so it must run with a running loop.
        self.manager.start()

    async def _worker(self, job: Job, token: CancellationToken) -> None:
        """Fake worker: stay Running until the test releases this Job.

        The releasing rule sets the Job's terminal status before signalling the
        gate, so ``_run_worker`` observes an already-terminal Job and just frees
        the slot it held (Req 7.5).
        """
        event = self._release.setdefault(job.id, asyncio.Event())
        await event.wait()

    async def _pump(self) -> None:
        """Let the dispatcher and any worker tasks make all available progress."""
        for _ in range(_PUMP_TURNS):
            await asyncio.sleep(0)

    def _running_ids(self) -> list[str]:
        """Ids of Jobs currently in status Running, in a stable order."""
        return sorted(
            jid
            for jid in self._active
            if self.manager.get(jid).status is JobStatus.RUNNING
        )

    @rule()
    def submit(self) -> None:
        """Submit a new Job and let the dispatcher consider promoting it."""
        self._counter += 1
        job = _make_job(f"svc-{self._counter}")
        # Pre-create the release gate so a completion can always find it even if
        # the worker has not started running yet.
        self._release[job.id] = asyncio.Event()
        # Inject directly into the manager's table/queue mirroring create_job,
        # keeping the dispatcher and semaphore as the real System under test.
        self.manager._jobs[job.id] = job
        self.manager._queue.append(job.id)
        self._active.add(job.id)
        self.manager.notify()
        self._loop.run_until_complete(self._pump())

    @precondition(lambda self: bool(self._running_ids()))
    @rule(
        idx=st.integers(min_value=0, max_value=1_000_000),
        status=st.sampled_from(_TERMINAL_STATUSES),
    )
    def complete(self, idx: int, status: JobStatus) -> None:
        """Drive one Running Job terminal and release it, freeing its slot."""
        running = self._running_ids()
        job_id = running[idx % len(running)]
        job = self.manager.get(job_id)

        running_before = self.manager.running_count

        # Terminal status is set before the gate is opened so the worker's
        # cleanup does not override it; the worker then returns and _run_worker
        # releases exactly the one slot this Job held (Req 7.5).
        job.status = status
        self._active.discard(job_id)
        self._release[job_id].set()
        self._loop.run_until_complete(self._pump())

        # The completed Job is no longer Running.
        assert self.manager.get(job_id).status is status
        # Exactly one slot was freed: relative to the moment before completion,
        # Running drops by one unless a single Queued Job took the freed slot.
        backlog_before = len(self._active) >= running_before
        expected_after = running_before if backlog_before else running_before - 1
        assert self.manager.running_count == expected_after, (
            f"completing {job_id} did not free exactly one slot: "
            f"running was {running_before}, now {self.manager.running_count}"
        )

    @invariant()
    def running_never_exceeds_limit_and_matches_capacity(self) -> None:
        """Running count tracks min(limit, active) at every observed point."""
        running = self.manager.running_count
        # Req 7.3 / 7.7: never more Running than the configured limit.
        assert running <= self.concurrency_limit, (
            f"running_count {running} exceeded limit {self.concurrency_limit}"
        )
        # Slots are fully utilised but never over-committed: Running equals the
        # number of non-terminal Jobs capped at the limit. This holds only once
        # a terminal transition has freed exactly one slot per completion.
        expected = min(self.concurrency_limit, len(self._active))
        assert running == expected, (
            f"running_count {running} != min(limit={self.concurrency_limit}, "
            f"active={len(self._active)})={expected}"
        )

    def teardown(self) -> None:
        """Release any blocked workers and tear down the loop."""
        try:
            for event in self._release.values():
                event.set()
            self._loop.run_until_complete(self._pump())
            self._loop.run_until_complete(self.manager.stop())
        finally:
            self._loop.close()


class _ConcurrencyLimit1Machine(_ConcurrencyMachine):
    """Serialized case: a single running slot."""

    concurrency_limit = 1


class _ConcurrencyLimit3Machine(_ConcurrencyMachine):
    """Multi-slot case: several Jobs may run concurrently with a backlog."""

    concurrency_limit = 3


# Feature: discord-ollama-coding-agent, Property 6
# Validates: Requirements 7.3, 7.5, 7.7
def test_running_jobs_never_exceed_concurrency_limit_one_slot() -> None:
    """Property 6 with a single running slot (strict serialization)."""
    run_state_machine_as_test(
        _ConcurrencyLimit1Machine,
        settings=settings(max_examples=20, deadline=None, stateful_step_count=15),
    )


# Feature: discord-ollama-coding-agent, Property 6
# Validates: Requirements 7.3, 7.5, 7.7
def test_running_jobs_never_exceed_concurrency_limit_multi_slot() -> None:
    """Property 6 with multiple running slots and a queue backlog."""
    run_state_machine_as_test(
        _ConcurrencyLimit3Machine,
        settings=settings(max_examples=20, deadline=None, stateful_step_count=15),
    )
