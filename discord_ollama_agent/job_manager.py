"""Job_Manager: Job identity, queue, concurrency, and lifecycle.

Owns the in-memory Job table, the FIFO queue, the concurrency semaphore, the
dispatcher loop, and all Job_Status transitions (including cancellation).

This module implements:

- :meth:`JobManager.create_job` -- mints a new Queued Job for a target with a
  unique, never-reused id of the form ``"{target_name}-{unique_suffix}"`` and
  enqueues it (Req 1.1, 1.3, 7.1).
- :meth:`JobManager.create_revision` -- mints a new Queued Job that continues the
  branch of a previously completed Job (Req 11.1, 11.10).
- :meth:`JobManager.get` -- looks a Job up by id (Req 6.4, 6.5, 8.2, 11.6).
- :meth:`JobManager._dispatch_loop` -- the consumer coroutine that, while a slot
  is free and a Queued Job exists, promotes the earliest-submission Queued Job
  to Running and spawns a worker :class:`asyncio.Task`, freeing the slot on every
  terminal transition (Req 7.3, 7.4, 7.5, 7.7).
- :meth:`JobManager.cancel` -- cancels a Job by id, with an outcome that depends
  only on its current status: Queued is removed from the queue and set Cancelled;
  Running is signalled to stop (terminating its build/test command) and set
  Cancelled; a terminal Job is rejected unchanged (Req 8.3, 8.4, 8.6).

Concurrency follows the design's single-event-loop model: an
:class:`asyncio.Semaphore` sized to the configured concurrency limit bounds the
number of concurrently Running Jobs, and a FIFO queue of Queued ids (in
submission order) feeds the dispatcher. The semaphore is the single gate, so a
slot is held for exactly the lifetime of a Running Job and released on its
terminal transition. Jobs are held in memory only and are **not** persisted
across restarts.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .execution_sandbox import CancellationToken
from .models.job import Job, JobEvent, JobStatus
from .models.target import RegisteredTarget

__all__ = [
    "JobManager",
    "CancelOutcome",
    "CancelResult",
    "Worker",
]

logger = logging.getLogger(__name__)

#: A Job worker: runs a single Job's lifecycle to a terminal state, observing the
#: supplied :class:`CancellationToken` so a cancelled Job stops without
#: committing or pushing. :meth:`CodingAgent.run` matches this signature.
Worker = Callable[[Job, CancellationToken], Awaitable[None]]

#: Optional async sink invoked with a :class:`JobEvent` on a status transition
#: the Job_Manager itself drives (currently the cancellation of a Queued Job,
#: which has no worker to emit the event).
JobEventSink = Callable[[JobEvent], Awaitable[None]]


class CancelResult(str, Enum):
    """The outcome category of a :meth:`JobManager.cancel` invocation.

    The result depends solely on the Job's status at the moment of the call
    (Property 30): a Queued or Running Job is cancelled, a terminal Job is
    rejected, and an unknown id is reported as not found. ``str``-backed so the
    value renders and compares cleanly.
    """

    NOT_FOUND = "not_found"
    CANCELLED_QUEUED = "cancelled_queued"
    CANCELLED_RUNNING = "cancelled_running"
    REJECTED_TERMINAL = "rejected_terminal"


@dataclass(frozen=True)
class CancelOutcome:
    """The result of attempting to cancel a Job.

    Carries the :class:`CancelResult` category and the resolved :class:`Job`
    (``None`` only when the id was not found), so the Discord_Bot can render the
    matching reply: a not-found message (Req 8.2), a cancellation confirmation
    referencing the Job id (Req 8.3, 8.4), or a terminal-state rejection
    (Req 8.6).
    """

    result: CancelResult
    job: Job | None = None

    @property
    def cancelled(self) -> bool:
        """Whether the Job was cancelled (it was Queued or Running)."""
        return self.result in (
            CancelResult.CANCELLED_QUEUED,
            CancelResult.CANCELLED_RUNNING,
        )

    @property
    def rejected_terminal(self) -> bool:
        """Whether cancellation was rejected because the Job is terminal (Req 8.6)."""
        return self.result is CancelResult.REJECTED_TERMINAL

    @property
    def not_found(self) -> bool:
        """Whether no Job with the requested id exists (Req 8.2)."""
        return self.result is CancelResult.NOT_FOUND


class JobManager:
    """Owns Job identity, the in-memory Job table, the FIFO queue, and dispatch.

    Job ids are minted with a process-wide-monotonic counter combined with the
    associated target name, so every id contains the target name, is unique for
    the System's lifetime, and is never reused even after a Job terminates
    (Req 7.1). New Jobs start in :attr:`JobStatus.QUEUED` and are appended to the
    FIFO queue in submission order; ``submitted_at`` is the tie-broken ordering
    key the dispatcher consumes (Req 7.4).

    Concurrency is bounded by an :class:`asyncio.Semaphore` sized to
    ``concurrency_limit`` (Req 7.6); the :meth:`_dispatch_loop` coroutine acquires
    a slot, promotes the earliest-submission Queued Job to Running, and spawns a
    worker :class:`asyncio.Task`. The slot is released when the worker reaches a
    terminal state, making exactly one slot available per terminal transition
    (Req 7.3, 7.5, 7.7).

    Args:
        concurrency_limit: Maximum number of concurrently Running Jobs (Req 7.6).
        worker: The coroutine that executes a single Job (e.g. ``CodingAgent.run``).
            Optional so the identity/queue methods can be exercised without an
            executor; :meth:`start` requires it.
        event_sink: Optional async callback invoked with a :class:`JobEvent` when
            the Job_Manager itself drives a transition (cancelling a Queued Job).
    """

    def __init__(
        self,
        concurrency_limit: int = 1,
        worker: Worker | None = None,
        event_sink: JobEventSink | None = None,
    ) -> None:
        # In-memory Job table keyed by Job id. Jobs are never persisted across
        # restarts (Req 7.1 lifetime semantics).
        self._jobs: dict[str, Job] = {}
        # FIFO queue of Job ids awaiting dispatch, in submission order (Req 7.4).
        self._queue: deque[str] = deque()
        # Monotonic suffix source. Strictly increasing and never reset, so a
        # suffix (and therefore a Job id) is never reused for the System's
        # lifetime (Req 7.1).
        self._suffix_counter = itertools.count(1)

        # Concurrency control (Req 7.3, 7.5, 7.6, 7.7).
        self._concurrency_limit = int(concurrency_limit)
        self._worker = worker
        self._event_sink = event_sink
        # The single gate bounding Running Jobs: one permit per running slot.
        self._semaphore = asyncio.Semaphore(self._concurrency_limit)
        # Set whenever new work is enqueued or a slot frees, to wake the
        # dispatcher when it is idle waiting for a Queued Job.
        self._wakeup = asyncio.Event()
        # Per-Running-Job cancellation tokens and worker tasks, so cancellation
        # can signal the right Job and so workers can be tracked/cleaned up.
        self._cancel_tokens: dict[str, CancellationToken] = {}
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}
        # The dispatcher task, created by ``start`` and cleared by ``stop``.
        self._dispatch_task: asyncio.Task[None] | None = None
        self._stopping = False

    def _next_suffix(self) -> int:
        """Return the next strictly-increasing, never-reused id suffix."""

        return next(self._suffix_counter)

    def create_job(
        self,
        target: RegisteredTarget,
        idea: str,
        channel_id: int,
        user_id: int,
    ) -> Job:
        """Create a new Queued Job for ``target`` and enqueue it.

        The Job id is ``"{target.name}-{unique_suffix}"`` where the suffix comes
        from a monotonic counter, so the id contains the target name and is
        unique and never reused for the System's lifetime (Req 1.1, 1.3, 7.1).
        The Job records its originating Discord ``channel_id`` and ``user_id`` and
        a ``submitted_at`` timestamp used as the FIFO ordering key (Req 7.4), then
        is stored in the in-memory table and appended to the queue.
        """

        job_id = f"{target.name}-{self._next_suffix()}"
        job = Job(
            id=job_id,
            target_name=target.name,
            idea=idea,
            status=JobStatus.QUEUED,
            submitted_at=datetime.now(timezone.utc),
            channel_id=channel_id,
            user_id=user_id,
        )
        self._jobs[job_id] = job
        self._queue.append(job_id)
        return job

    def create_revision(
        self,
        base_job: Job,
        feedback: str,
        channel_id: int,
        user_id: int,
    ) -> Job:
        """Create a new Queued Job that continues ``base_job``'s branch.

        Mints a fresh, unique, never-reused id for the same target as
        ``base_job`` and produces a new Job that continues work on
        ``base_job.branch_name`` (Req 11.1). The new Job is marked
        ``is_revision=True`` and links back via ``base_job_id``; ``feedback`` is
        carried as the Job's idea so the Coding_Agent can load the existing
        branch contents plus the feedback as generation context. The revision is
        subject to the same queue and concurrency rules as a normal Job
        (Req 11.10).
        """

        job_id = f"{base_job.target_name}-{self._next_suffix()}"
        job = Job(
            id=job_id,
            target_name=base_job.target_name,
            idea=feedback,
            status=JobStatus.QUEUED,
            submitted_at=datetime.now(timezone.utc),
            channel_id=channel_id,
            user_id=user_id,
            branch_name=base_job.branch_name,
            is_revision=True,
            base_job_id=base_job.id,
        )
        self._jobs[job_id] = job
        self._queue.append(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the Job with ``job_id`` or ``None`` if no such Job exists.

        Used by ``/status``, ``/cancel``, and ``/revise`` to resolve a Job id and
        reply with a not-found message when it is absent (Req 6.4, 6.5, 8.2,
        11.6).
        """

        return self._jobs.get(job_id)

    # -- Dispatch / concurrency ------------------------------------------------

    @property
    def concurrency_limit(self) -> int:
        """The configured maximum number of concurrently Running Jobs (Req 7.6)."""
        return self._concurrency_limit

    @property
    def running_count(self) -> int:
        """The number of Jobs currently Running (each holds one slot)."""
        return len(self._worker_tasks)

    def notify(self) -> None:
        """Wake the dispatcher to (re)consider the queue.

        Callers enqueue work via :meth:`create_job`/:meth:`create_revision` (which
        stay free of any event-loop dependency) and then call ``notify`` so the
        dispatcher, if idle, picks the new Job up. Idempotent and safe to call
        with no running loop (the wakeup is a plain flag).
        """
        self._wakeup.set()

    def start(self) -> None:
        """Start the dispatcher coroutine if it is not already running.

        Must be called from within the running event loop. Requires a ``worker``
        to have been supplied at construction; otherwise there is nothing to
        execute promoted Jobs.
        """
        if self._worker is None:
            raise RuntimeError("JobManager.start requires a worker callable")
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        self._stopping = False
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        """Stop the dispatcher and wait for it to unwind.

        Signals the loop to exit and wakes it if idle. In-flight worker tasks are
        left to finish on their own; this only tears down the dispatcher.
        """
        self._stopping = True
        self._wakeup.set()
        task = self._dispatch_task
        self._dispatch_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _dispatch_loop(self) -> None:
        """Promote Queued Jobs to Running as slots free, in submission order.

        Each iteration waits for the next Queued Job (FIFO by submission time;
        Req 7.4), then acquires a running slot from the semaphore (waiting while
        the System is at the concurrency limit; Req 7.3). Only once a slot is
        held is the Job transitioned to Running and a worker task spawned, so the
        number of Running Jobs never exceeds the limit (Req 7.7). A Job cancelled
        while it waited for a slot is skipped and the slot returned.
        """
        while not self._stopping:
            job = await self._take_next_queued()
            if job is None:
                # Stopping with no work to dispatch.
                break

            await self._semaphore.acquire()

            # The Job may have been cancelled while it waited for a slot; if so,
            # return the slot and move on without running it.
            if job.status is not JobStatus.QUEUED:
                self._semaphore.release()
                continue

            self._spawn_worker(job)

    def _spawn_worker(self, job: Job) -> None:
        """Transition ``job`` to Running and launch its worker task (Req 7.4).

        A fresh :class:`CancellationToken` is created and retained so
        :meth:`cancel` can signal this specific Job; the worker runs under
        :meth:`_run_worker`, which releases the held slot on the terminal
        transition (Req 7.5).
        """
        assert self._worker is not None  # guaranteed by ``start``
        token = CancellationToken()
        self._cancel_tokens[job.id] = token
        job.status = JobStatus.RUNNING
        task = asyncio.create_task(self._run_worker(job, token))
        self._worker_tasks[job.id] = task

    async def _run_worker(self, job: Job, token: CancellationToken) -> None:
        """Run a Job's worker to a terminal state, then free its slot.

        On any unexpected error the Job is forced to a terminal Failed state so a
        slot is never leaked; whatever the outcome, the held semaphore slot is
        released exactly once (making one slot available per terminal transition;
        Req 7.5) and the dispatcher is woken to consider the next Queued Job.
        """
        try:
            await self._worker(job, token)
        except asyncio.CancelledError:
            if not job.status.is_terminal:
                job.status = JobStatus.CANCELLED
            raise
        except Exception:  # pragma: no cover - defensive; worker owns its errors
            logger.exception("Job %s worker raised; forcing Failed", job.id)
            if not job.status.is_terminal:
                job.status = JobStatus.FAILED
                job.failure_reason = job.failure_reason or "internal-error"
        finally:
            self._cancel_tokens.pop(job.id, None)
            self._worker_tasks.pop(job.id, None)
            # Free the slot this Job held and re-pump the dispatcher (Req 7.5).
            self._semaphore.release()
            self._wakeup.set()

    async def _take_next_queued(self) -> Job | None:
        """Return the earliest-submission Queued Job, waiting if none is ready.

        Pops ids from the front of the FIFO queue (submission order), discarding
        any that are no longer Queued (e.g. cancelled while queued), so the Job
        returned is always the Queued Job with the earliest submission time
        (Req 7.4). When the queue holds no Queued Job the coroutine waits for a
        wakeup; it returns ``None`` once the manager is stopping.
        """
        while not self._stopping:
            job = self._pop_next_queued()
            if job is not None:
                return job
            # Clear then re-check to avoid missing a wakeup raced in between.
            self._wakeup.clear()
            job = self._pop_next_queued()
            if job is not None:
                return job
            if self._stopping:
                break
            await self._wakeup.wait()
        return None

    def _pop_next_queued(self) -> Job | None:
        """Remove and return the front Queued Job, discarding stale ids.

        Ids whose Job is missing or no longer Queued are dropped from the queue.
        Returns ``None`` when no Queued Job remains.
        """
        while self._queue:
            job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            if job is not None and job.status is JobStatus.QUEUED:
                return job
        return None

    # -- Cancellation ----------------------------------------------------------

    async def cancel(self, job_id: str) -> CancelOutcome:
        """Cancel the Job ``job_id``; the outcome depends only on its status.

        - **Queued**: the Job is removed from the FIFO queue and set Cancelled,
          so the dispatcher never promotes it (Req 8.3).
        - **Running**: the Job's :class:`CancellationToken` is signalled (which
          terminates any active build/test command in the sandbox) and the Job is
          set Cancelled; its worker observes the signal and stops without
          committing or pushing (Req 8.4, 8.5). The held slot is freed when the
          worker unwinds (Req 7.5).
        - **Succeeded/Failed/Cancelled**: rejected, leaving the status unchanged
          (Req 8.6).
        - **Unknown id**: reported as not found (Req 8.2).

        This method performs its status read and mutation without awaiting, so on
        the single event loop it is atomic with respect to the worker tasks.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return CancelOutcome(CancelResult.NOT_FOUND, None)

        status = job.status
        if status is JobStatus.QUEUED:
            self._remove_from_queue(job_id)
            job.status = JobStatus.CANCELLED
            # No worker exists for a Queued Job, so emit the transition here.
            await self._emit(job)
            return CancelOutcome(CancelResult.CANCELLED_QUEUED, job)

        if status is JobStatus.RUNNING:
            token = self._cancel_tokens.get(job_id)
            if token is not None:
                # Signals the sandbox to terminate the running build/test command
                # (Req 8.4); the worker then stops without committing (Req 8.5).
                token.cancel()
            job.status = JobStatus.CANCELLED
            # The worker emits the Cancelled event as it unwinds; avoid a double
            # emission here.
            return CancelOutcome(CancelResult.CANCELLED_RUNNING, job)

        # Terminal: Succeeded, Failed, or already Cancelled (Req 8.6).
        return CancelOutcome(CancelResult.REJECTED_TERMINAL, job)

    def _remove_from_queue(self, job_id: str) -> None:
        """Remove ``job_id`` from the FIFO queue if present (Req 8.3)."""
        try:
            self._queue.remove(job_id)
        except ValueError:
            pass

    async def _emit(self, job: Job) -> None:
        """Emit a :class:`JobEvent` snapshot of ``job`` to the sink, if any."""
        if self._event_sink is None:
            return
        await self._event_sink(JobEvent.from_job(job))
