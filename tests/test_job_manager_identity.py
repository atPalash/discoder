"""Property-based test for Job identity (task 10.2).

Covers Property 4: Job identity contains the target name, is unique, and is
never reused.

*For any* sequence of Job creations -- via :meth:`JobManager.create_job` and
:meth:`JobManager.create_revision`, interleaved with marking previously created
Jobs terminal and reusing target names -- every minted Job id contains its
associated target's name, and the set of all ids ever produced is pairwise
unique with no id ever reused.

The sequence deliberately mixes:
  - repeated use of the same target names (so uniqueness cannot come from the
    target name alone);
  - revisions built on earlier Jobs, including Jobs that have been driven to a
    terminal state (so id minting after terminal transitions is exercised);
  - target names that overlap with suffix-like fragments, to make sure the
    "id contains target name" check is meaningful and uniqueness still holds.

**Validates: Requirements 1.3, 7.1**
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.job import JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

# A small pool of valid target names. Reuse across operations is intentional so
# many Jobs share a target name and uniqueness must come from the suffix, not
# the name. Names that overlap with numeric/suffix fragments (e.g. "svc-1") are
# included so an id like "svc-1-<suffix>" still passes the containment check and
# never collides with a future "svc-1" Job's id.
_TARGET_NAMES = ["svc", "api", "svc-1", "a", "data-pipeline", "x"]

# Terminal states a Job may be driven to between creations, to exercise id
# minting after terminal transitions (Property 4 is explicit about this).
_TERMINAL_STATUSES = [
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
]


def _make_target(name: str) -> RegisteredTarget:
    """Build a valid target whose only meaningful variable here is ``name``."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("workspace") / "t",
        repo_remote="git@example.com:org/repo.git",
        credentials_ref="GIT_TOKEN",
    )


# Each operation is one of:
#   ("create", target_name)        -> create_job for that target
#   ("revise", selector)           -> create_revision on an existing job
#   ("terminal", selector, status) -> mark an existing job terminal
# Selectors are arbitrary ints reduced modulo the live job count at apply time,
# so an operation always targets a real job whenever one exists.
_OP = st.one_of(
    st.tuples(st.just("create"), st.sampled_from(_TARGET_NAMES)),
    st.tuples(st.just("revise"), st.integers(min_value=0, max_value=1_000_000)),
    st.tuples(
        st.just("terminal"),
        st.integers(min_value=0, max_value=1_000_000),
        st.sampled_from(_TERMINAL_STATUSES),
    ),
)


# Feature: discord-ollama-coding-agent, Property 4: Job identity contains the target name, is unique, and is never reused
# Validates: Requirements 1.3, 7.1
@settings(max_examples=20, deadline=None)
@given(operations=st.lists(_OP, min_size=1, max_size=60))
def test_job_identity_contains_target_name_unique_and_never_reused(operations):
    """Every minted id contains its target name and all ids are unique."""
    manager = JobManager()
    targets = {name: _make_target(name) for name in _TARGET_NAMES}

    created_jobs = []  # every Job ever created, in creation order
    seen_ids = set()  # all ids ever minted; used to detect reuse/collision
    channel_id = 100
    user_id = 7

    def _record(job, expected_target_name):
        # Req 1.3 / 7.1: the id contains the associated target's name.
        assert expected_target_name in job.id, (
            f"id {job.id!r} does not contain target name "
            f"{expected_target_name!r}"
        )
        # Req 7.1: never reused -- this id must not have been minted before.
        assert job.id not in seen_ids, f"id {job.id!r} was reused"
        seen_ids.add(job.id)
        created_jobs.append(job)

    for op in operations:
        kind = op[0]

        if kind == "create":
            target_name = op[1]
            job = manager.create_job(
                targets[target_name],
                idea="add a healthcheck endpoint",
                channel_id=channel_id,
                user_id=user_id,
            )
            _record(job, target_name)

        elif kind == "revise":
            if not created_jobs:
                continue
            base = created_jobs[op[1] % len(created_jobs)]
            job = manager.create_revision(
                base,
                feedback="please tweak the output",
                channel_id=channel_id,
                user_id=user_id,
            )
            # A revision keeps the base Job's target, so the new id must
            # contain that same target name (Req 1.3, 7.1).
            _record(job, base.target_name)

        else:  # "terminal"
            if not created_jobs:
                continue
            job = created_jobs[op[1] % len(created_jobs)]
            # Drive the Job to a terminal state so subsequent creations are
            # exercised "after other Jobs reach terminal states" (Property 4).
            job.status = op[2]

    # Pairwise uniqueness across every id ever produced (no reuse over the
    # whole sequence, including after terminal transitions).
    all_ids = [job.id for job in created_jobs]
    assert len(all_ids) == len(set(all_ids)), "Job ids are not pairwise unique"
