"""Job data models and lifecycle types for the Discord-Ollama Coding Agent.

These pydantic v2 models describe a unit of work and the events emitted as it
moves through its lifecycle:

- :class:`JobStatus` -- the lifecycle states a Job moves through. Backed by
  ``str`` so values serialize as their human-readable names and compare cleanly
  against plain strings.
- :class:`Job`       -- a single submitted idea (or revision) and all state the
  orchestration layer tracks for it: identity, originating Discord context,
  attempt count, branch name, and terminal outcome details.
- :class:`JobEvent`  -- the payload emitted on each status transition and handed
  to the Discord event sink so it can post started/succeeded/failed messages to
  the originating channel (Req 6.1-6.3).

Jobs are held in memory by the Job_Manager and are **not** persisted across
restarts. Job identifiers contain the target name, are unique for the System's
lifetime, and are never reused (Req 7.1).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

__all__ = [
    "JobStatus",
    "Job",
    "JobEvent",
]


class JobStatus(str, Enum):
    """The lifecycle states a Job moves through.

    ``QUEUED`` and ``RUNNING`` are non-terminal; ``SUCCEEDED``, ``FAILED``, and
    ``CANCELLED`` are terminal. A Job only ever leaves a terminal state via
    ``/revise``, which creates a *new* Job rather than mutating the existing one.

    The enum is ``str``-backed so members serialize as their string values
    (e.g. ``"Queued"``) and compare equal to those strings, which keeps
    persistence and message rendering straightforward.
    """

    QUEUED = "Queued"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether this status is terminal (no further automatic transitions).

        The orchestration layer uses this to decide when to free a running slot
        and to reject operations such as ``/cancel`` against finished Jobs
        (Req 7.5, 8.6).
        """

        return self in _TERMINAL_STATUSES


# Defined once and referenced by ``JobStatus.is_terminal`` to avoid rebuilding
# the set on every access.
_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class Job(BaseModel):
    """A single submitted idea (or revision) tracked through its lifecycle.

    ``id`` is composed of the target name plus a unique component, is unique for
    the System's lifetime, and is never reused (Req 7.1). ``submitted_at`` is the
    FIFO ordering key the dispatcher uses to pick the next Queued Job (Req 7.4).
    ``channel_id`` records the originating Discord channel so status messages are
    posted back to the right place (Req 6.1-6.3).

    ``branch_name`` is retained after success so ``/revise`` can continue work on
    the same branch; its absence is what ``/revise`` checks when deciding a Job
    can no longer be revised (Req 11.1, 11.9). ``is_revision`` and
    ``base_job_id`` mark a Job created via ``/revise`` and link it back to the
    Job it continues (Req 11.1).
    """

    id: str
    target_name: str
    idea: str
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime
    channel_id: int
    user_id: int
    attempts_completed: int = 0
    branch_name: str | None = None
    failure_reason: str | None = None
    result_note: str | None = None
    is_revision: bool = False
    base_job_id: str | None = None


class JobEvent(BaseModel):
    """Payload emitted on a Job status transition for the Discord event sink.

    Carries everything :meth:`DiscordBot.on_job_event` needs to render and post a
    message without re-fetching the Job: the Job ``id`` and the ``status`` it
    transitioned to, the originating ``channel_id`` so the message lands in the
    right channel (Req 6.1-6.3), and the relevant outcome details for that
    transition:

    - ``branch_name`` / ``result_note`` populate the Succeeded completion message
      (the pushed branch name or a no-changes note) (Req 6.2).
    - ``failure_reason`` populates the Failed message (Req 6.3).

    The detail fields are optional because which ones are meaningful depends on
    the ``status`` (e.g. a Running event carries none of them).
    """

    job_id: str
    status: JobStatus
    channel_id: int
    branch_name: str | None = None
    result_note: str | None = None
    failure_reason: str | None = None

    @classmethod
    def from_job(cls, job: Job) -> "JobEvent":
        """Build a :class:`JobEvent` reflecting a Job's current state.

        Convenience for the orchestration layer: snapshots the fields the event
        sink needs from ``job`` at the moment of a transition.
        """

        return cls(
            job_id=job.id,
            status=job.status,
            channel_id=job.channel_id,
            branch_name=job.branch_name,
            result_note=job.result_note,
            failure_reason=job.failure_reason,
        )
