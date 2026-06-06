"""Property-based test for status message content (task 12.6).

Covers Property 20: Status messages carry the Job identifier and correct state
information.

*For any* Job, the rendered started/succeeded/failed message and the ``/status``
reply contain the Job identifier and the corresponding state information:
Succeeded includes the pushed branch name or a no-changes note; Failed includes
the recorded failure reason; ``/status`` includes the current Job_Status.

**Validates: Requirements 6.2, 6.3, 6.4, 11.11**

These tests drive the real status-rendering surfaces of :class:`DiscordBot`:

- The lifecycle event sink
  (:meth:`~discord_ollama_agent.discord_bot.DiscordBot.on_job_event`) which
  renders the started/succeeded/failed message and posts it through a
  :class:`~discord_ollama_agent.discord_bot.MessageSink`; a fake sink captures
  the posted content (Req 6.2, 6.3, 11.11).
- The ``/status`` handler
  (:meth:`~discord_ollama_agent.discord_bot.DiscordBot.on_status`) wired to a
  real :class:`~discord_ollama_agent.job_manager.JobManager`; the reply must
  carry the Job id and the Job's current status (Req 6.4).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.discord_bot import DiscordBot
from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.models.job import Job, JobEvent, JobStatus

ALLOWED_CHANNEL = 1000
USER = 10
ADMIN = 20


@dataclass(frozen=True)
class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    channel_id: int
    user_id: int


@dataclass
class _CapturingSink:
    """A fake MessageSink that records every posted (channel, content) pair.

    Delivery always succeeds, so :meth:`DiscordBot.on_job_event` posts exactly
    one message per renderable transition and the captured content is the
    rendered status message under test (Req 6.2, 6.3, 11.11).
    """

    posts: list[tuple[int, str]] = field(default_factory=list)

    async def send(self, channel_id: int, content: str) -> None:
        self.posts.append((channel_id, content))


def _make_bot(sink: _CapturingSink | None = None) -> tuple[DiscordBot, JobManager]:
    """Build a DiscordBot wired to a real JobManager and (optionally) a sink."""
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[ALLOWED_CHANNEL],
        authorized_users=[USER],
        admin_users=[ADMIN],
    )
    manager = JobManager()
    bot = DiscordBot(config, job_manager=manager, message_sink=sink)
    return bot, manager


# Job identifiers contain the target name plus a unique suffix (Req 7.1); model
# that shape with a non-empty target-ish prefix and an integer suffix. Text is
# kept printable so substring containment in the rendered message is meaningful.
_job_ids = st.builds(
    lambda prefix, suffix: f"{prefix}-{suffix}",
    st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126),
        min_size=1,
        max_size=20,
    ),
    st.integers(min_value=1, max_value=10_000),
)

# Branch names and free-form detail text for Succeeded/Failed events. Non-empty
# so they are distinguishable from the no-changes default and always assertable.
_detail_text = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)


# Feature: discord-ollama-coding-agent, Property 20
# Property 20: Status messages carry the Job identifier and correct state information.
# Validates: Requirements 6.2, 6.3, 6.4, 11.11
@settings(max_examples=20, deadline=None)
@given(job_id=_job_ids)
def test_started_message_carries_job_id(job_id: str) -> None:
    """A Running transition posts a started message referencing the Job id (Req 6.1)."""
    sink = _CapturingSink()
    bot, _ = _make_bot(sink)
    event = JobEvent(job_id=job_id, status=JobStatus.RUNNING, channel_id=ALLOWED_CHANNEL)

    asyncio.run(bot.on_job_event(event))

    # Exactly one message was posted to the originating channel.
    assert len(sink.posts) == 1
    channel_id, content = sink.posts[0]
    assert channel_id == ALLOWED_CHANNEL
    # The started message references the Job id and indicates the started state.
    assert job_id in content
    assert "started" in content.lower()


# Feature: discord-ollama-coding-agent, Property 20
# Property 20: Status messages carry the Job identifier and correct state information.
# Validates: Requirements 6.2, 6.3, 6.4, 11.11
@settings(max_examples=20, deadline=None)
@given(
    job_id=_job_ids,
    # Either a pushed branch name (non-empty branch_name) or a no-changes
    # outcome (no branch; an optional explicit result note) (Req 6.2).
    branch_name=st.one_of(st.none(), _detail_text),
    result_note=st.one_of(st.none(), _detail_text),
)
def test_succeeded_message_carries_job_id_and_outcome(
    job_id: str, branch_name: str | None, result_note: str | None
) -> None:
    """A Succeeded transition posts the Job id plus the branch name or a
    no-changes note (Req 6.2, 11.11)."""
    sink = _CapturingSink()
    bot, _ = _make_bot(sink)
    event = JobEvent(
        job_id=job_id,
        status=JobStatus.SUCCEEDED,
        channel_id=ALLOWED_CHANNEL,
        branch_name=branch_name,
        result_note=result_note,
    )

    asyncio.run(bot.on_job_event(event))

    assert len(sink.posts) == 1
    channel_id, content = sink.posts[0]
    assert channel_id == ALLOWED_CHANNEL
    # The completion message references the Job id and the success state.
    assert job_id in content
    assert "succeeded" in content.lower()

    if branch_name:
        # The pushed branch name appears in the message (Req 6.2).
        assert branch_name in content
    elif result_note:
        # With no branch, the explicit no-changes note appears (Req 6.2).
        assert result_note in content
    else:
        # With neither, a generic no-changes note is rendered (Req 6.2).
        assert "no changes" in content.lower()


# Feature: discord-ollama-coding-agent, Property 20
# Property 20: Status messages carry the Job identifier and correct state information.
# Validates: Requirements 6.2, 6.3, 6.4, 11.11
@settings(max_examples=20, deadline=None)
@given(
    job_id=_job_ids,
    failure_reason=st.one_of(st.none(), _detail_text),
)
def test_failed_message_carries_job_id_and_reason(
    job_id: str, failure_reason: str | None
) -> None:
    """A Failed transition posts the Job id plus the recorded failure reason (Req 6.3)."""
    sink = _CapturingSink()
    bot, _ = _make_bot(sink)
    event = JobEvent(
        job_id=job_id,
        status=JobStatus.FAILED,
        channel_id=ALLOWED_CHANNEL,
        failure_reason=failure_reason,
    )

    asyncio.run(bot.on_job_event(event))

    assert len(sink.posts) == 1
    channel_id, content = sink.posts[0]
    assert channel_id == ALLOWED_CHANNEL
    # The failure message references the Job id and the failed state.
    assert job_id in content
    assert "failed" in content.lower()

    if failure_reason:
        # The recorded failure reason appears verbatim (Req 6.3).
        assert failure_reason in content
    else:
        # Absent a recorded reason, a generic failure reason is rendered (Req 6.3).
        assert "unknown failure" in content.lower()


# Feature: discord-ollama-coding-agent, Property 20
# Property 20: Status messages carry the Job identifier and correct state information.
# Validates: Requirements 6.2, 6.3, 6.4, 11.11
@settings(max_examples=20, deadline=None)
@given(
    job_id=_job_ids,
    status=st.sampled_from(list(JobStatus)),
)
def test_status_reply_carries_job_id_and_current_status(
    job_id: str, status: JobStatus
) -> None:
    """``/status`` replies with the Job id and the Job's current status (Req 6.4)."""
    bot, manager = _make_bot()
    job = Job(
        id=job_id,
        target_name=job_id.rsplit("-", 1)[0],
        idea="add a healthcheck endpoint",
        status=status,
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=ALLOWED_CHANNEL,
        user_id=USER,
    )
    manager._jobs[job.id] = job

    reply = asyncio.run(bot.on_status(_Interaction(ALLOWED_CHANNEL, USER), job_id))

    # The reply references the queried Job id and reports its current status.
    assert job_id in reply
    assert status.value in reply
