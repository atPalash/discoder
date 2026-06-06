"""Unit tests for the Job model and lifecycle types (task 2.3).

Covers JobStatus values and terminal classification, Job construction with
defaults, and JobEvent payload construction including JobEvent.from_job.
"""

from datetime import datetime, timezone

from discord_ollama_agent.models.job import Job, JobEvent, JobStatus


def _make_job(**overrides) -> Job:
    base = dict(
        id="svc-1",
        target_name="svc",
        idea="add a healthcheck endpoint",
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=42,
        user_id=7,
    )
    base.update(overrides)
    return Job(**base)


class TestJobStatus:
    def test_values_are_human_readable_strings(self):
        assert JobStatus.QUEUED.value == "Queued"
        assert JobStatus.RUNNING.value == "Running"
        assert JobStatus.SUCCEEDED.value == "Succeeded"
        assert JobStatus.FAILED.value == "Failed"
        assert JobStatus.CANCELLED.value == "Cancelled"

    def test_str_enum_compares_equal_to_plain_string(self):
        # str-backed enum so persistence/rendering can treat it as a string.
        assert JobStatus.QUEUED == "Queued"

    def test_terminal_states(self):
        assert JobStatus.SUCCEEDED.is_terminal is True
        assert JobStatus.FAILED.is_terminal is True
        assert JobStatus.CANCELLED.is_terminal is True

    def test_non_terminal_states(self):
        assert JobStatus.QUEUED.is_terminal is False
        assert JobStatus.RUNNING.is_terminal is False


class TestJob:
    def test_defaults(self):
        job = _make_job()
        assert job.status is JobStatus.QUEUED
        assert job.attempts_completed == 0
        assert job.branch_name is None
        assert job.failure_reason is None
        assert job.result_note is None
        assert job.is_revision is False
        assert job.base_job_id is None

    def test_required_fields_round_trip(self):
        job = _make_job()
        assert job.id == "svc-1"
        assert job.target_name == "svc"
        assert job.idea == "add a healthcheck endpoint"
        assert job.channel_id == 42
        assert job.user_id == 7
        assert job.submitted_at == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_revision_fields(self):
        job = _make_job(
            id="svc-2",
            is_revision=True,
            base_job_id="svc-1",
            branch_name="agent/svc-1",
        )
        assert job.is_revision is True
        assert job.base_job_id == "svc-1"
        assert job.branch_name == "agent/svc-1"

    def test_id_contains_target_name(self):
        # Req 7.1: the Job id contains the associated target name.
        job = _make_job(id="svc-abc123", target_name="svc")
        assert job.target_name in job.id


class TestJobEvent:
    def test_construct_succeeded_event_with_branch(self):
        event = JobEvent(
            job_id="svc-1",
            status=JobStatus.SUCCEEDED,
            channel_id=42,
            branch_name="agent/svc-1",
        )
        assert event.job_id == "svc-1"
        assert event.status is JobStatus.SUCCEEDED
        assert event.channel_id == 42
        assert event.branch_name == "agent/svc-1"
        assert event.failure_reason is None

    def test_construct_failed_event_with_reason(self):
        event = JobEvent(
            job_id="svc-1",
            status=JobStatus.FAILED,
            channel_id=42,
            failure_reason="max-attempts-exhausted",
        )
        assert event.status is JobStatus.FAILED
        assert event.failure_reason == "max-attempts-exhausted"

    def test_from_job_snapshots_relevant_fields(self):
        job = _make_job(
            id="svc-9",
            status=JobStatus.SUCCEEDED,
            channel_id=99,
            branch_name="agent/svc-9",
            result_note="branch pushed",
        )
        event = JobEvent.from_job(job)
        assert event.job_id == "svc-9"
        assert event.status is JobStatus.SUCCEEDED
        assert event.channel_id == 99
        assert event.branch_name == "agent/svc-9"
        assert event.result_note == "branch pushed"
        assert event.failure_reason is None
