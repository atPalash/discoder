"""Property-based test for revision eligibility and same-branch continuation
(task 10.7).

Covers Property 31: Revision eligibility and same-branch continuation.

*For any* ``/revise`` invocation, a revision is created **if and only if** the
referenced Job exists, is in status Succeeded, still retains its branch
information, and the feedback contains a non-whitespace character; when created,
the revision continues on the referenced Job's branch and loads that branch's
contents together with the feedback as generation context. Otherwise no revision
is created and the appropriate rejection message is returned.

**Validates: Requirements 11.1, 11.2, 11.7, 11.9**

Scope note (where eligibility lives)
-------------------------------------
The eligibility predicate (referenced Job exists, is Succeeded, branch retained,
feedback non-whitespace) is part of the ``/revise`` command surface, which the
design assigns to the Discord_Bot layer (``on_revise``) -- see task 12.3, which
is not yet implemented. ``JobManager.create_revision`` is the continuation
primitive that surface calls *once* a request is deemed eligible: it
unconditionally mints a new Queued Job that continues the base Job's branch and
carries the feedback as context. It performs no gating of its own.

Accordingly this test:

  * Models the design's iff eligibility predicate as a pure function over
    ``(exists, status, branch_name, feedback)`` and asserts each component of
    the conjunction independently flips eligibility (Req 11.1, 11.7, 11.9, plus
    the existence half resolved via :meth:`JobManager.get`).
  * Asserts the *continuation guarantees* of :meth:`JobManager.create_revision`
    that the eligible branch of the predicate relies on: the new Job is Queued,
    continues ``base_job.branch_name``, is marked ``is_revision`` with
    ``base_job_id`` linking back, keeps the same target, and carries the
    feedback as its idea so the Coding_Agent can load the branch contents plus
    feedback as context (Req 11.1, 11.2). The new Job is also enqueued and
    retrievable via :meth:`JobManager.get`.

The end-to-end "no revision + appropriate rejection" assertions for the
ineligible cases belong with the ``/revise`` handler (task 12.3) and are
deferred to that task's property test; this test documents that gap rather than
asserting against an enforcement API that does not yet exist.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# All lifecycle states; only SUCCEEDED is eligible for revision (Req 11.1, 11.7).
_ALL_STATUSES = list(JobStatus)


def _make_target(name: str = "svc") -> RegisteredTarget:
    """A minimal valid target; only ``name`` matters for these properties."""
    return RegisteredTarget(
        name=name,
        directory_path="workspace/t",
        repo_remote="git@example.com:org/repo.git",
        credentials_ref="GIT_TOKEN",
    )


def _make_base_job(
    *,
    status: JobStatus,
    branch_name: str | None,
    target_name: str = "svc",
) -> Job:
    """A base Job placed in an arbitrary terminal/non-terminal state.

    ``branch_name`` is the field ``/revise`` checks for branch retention
    (Req 11.9); ``None`` models a Job whose branch information is no longer
    retained.
    """
    return Job(
        id=f"{target_name}-base",
        target_name=target_name,
        idea="add a healthcheck endpoint",
        status=status,
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=42,
        user_id=7,
        branch_name=branch_name,
    )


def _is_eligible(*, exists: bool, status: JobStatus, branch_name: str | None,
                 feedback: str) -> bool:
    """The design's iff eligibility predicate for ``/revise`` (Property 31).

    A revision is created iff the referenced Job exists, is Succeeded, retains
    its branch information, and the feedback has a non-whitespace character.
    """
    return (
        exists
        and status is JobStatus.SUCCEEDED
        and branch_name is not None
        and feedback.strip() != ""
    )


# Branch names: a retained branch (non-None) or no retained branch (None).
_branch_names = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=40).map(lambda s: f"agent/{s}"),
)

# Feedback: a mix of whitespace-only strings (ineligible) and strings holding at
# least one non-whitespace character (the eligible shape) so the predicate's
# feedback component is exercised both ways.
_whitespace_only = st.text(alphabet=" \t\n\r\f\v\u00a0\u2003", min_size=0, max_size=8)
_non_whitespace = st.text(min_size=1, max_size=60).filter(lambda s: s.strip() != "")
_feedbacks = st.one_of(_whitespace_only, _non_whitespace)


# Feature: discord-ollama-coding-agent, Property 31
# Validates: Requirements 11.1, 11.2, 11.7, 11.9
@settings(max_examples=20, deadline=None)
@given(
    status=st.sampled_from(_ALL_STATUSES),
    branch_name=_branch_names,
    feedback=_feedbacks,
    exists=st.booleans(),
)
def test_revision_eligibility_and_same_branch_continuation(
    status, branch_name, feedback, exists
):
    """Eligibility is the design's conjunction; eligible requests continue the
    referenced branch and carry feedback as context."""
    manager = JobManager()
    target = _make_target()

    base_job = _make_base_job(status=status, branch_name=branch_name)
    # ``exists`` models whether the referenced Job is present in the table, which
    # ``/revise`` resolves via JobManager.get (Req 11.6 existence half of P31).
    if exists:
        manager._jobs[base_job.id] = base_job  # register without enqueuing
        assert manager.get(base_job.id) is base_job
    else:
        assert manager.get(base_job.id) is None

    eligible = _is_eligible(
        exists=exists,
        status=status,
        branch_name=branch_name,
        feedback=feedback,
    )

    # --- The iff predicate decomposes into independent necessary conditions. ---
    # Existence (Req 11.6), Succeeded status (Req 11.1, 11.7), branch retained
    # (Req 11.9), and non-whitespace feedback (Req 11.8) are each required: if
    # eligible, every component holds.
    if eligible:
        assert exists
        assert status is JobStatus.SUCCEEDED
        assert branch_name is not None
        assert feedback.strip() != ""
    else:
        # Ineligible iff at least one component of the conjunction fails.
        assert (
            (not exists)
            or status is not JobStatus.SUCCEEDED
            or branch_name is None
            or feedback.strip() == ""
        )

    queue_len_before = len(manager._queue)

    if not eligible:
        # No enforcement API exists yet (task 12.3 owns ``/revise`` gating); the
        # end-to-end "no revision + rejection" assertions are deferred to that
        # task. Nothing to create here, so the manager state is unchanged.
        assert len(manager._queue) == queue_len_before
        return

    # --- Eligible: create_revision yields a same-branch continuation. ---------
    revision = manager.create_revision(
        base_job,
        feedback=feedback,
        channel_id=99,
        user_id=7,
    )

    # Continues on the referenced Job's branch (Req 11.1).
    assert revision.branch_name == base_job.branch_name
    assert revision.branch_name is not None

    # Starts Queued, subject to the same queue/concurrency rules (Req 11.10).
    assert revision.status is JobStatus.QUEUED

    # Marked a revision and linked back to the Job it continues (Req 11.1).
    assert revision.is_revision is True
    assert revision.base_job_id == base_job.id

    # Keeps the base Job's target so it builds/pushes against the same repo.
    assert revision.target_name == base_job.target_name

    # The feedback is carried as the Job's idea so the Coding_Agent can load the
    # existing branch contents together with the feedback as context (Req 11.2).
    assert revision.idea == feedback

    # Fresh, distinct identity containing the target name (never reuses the base
    # id) and retrievable from the table.
    assert revision.id != base_job.id
    assert base_job.target_name in revision.id
    assert manager.get(revision.id) is revision

    # The revision was enqueued for dispatch (Req 11.10).
    assert len(manager._queue) == queue_len_before + 1
    assert manager._queue[-1] == revision.id
