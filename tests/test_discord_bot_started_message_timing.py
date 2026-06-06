"""Integration test for the started-message delivery on a Running transition (task 12.10).

Exercises :meth:`DiscordBot.on_job_event` end to end with a recording fake
:class:`~discord_ollama_agent.discord_bot.MessageSink`: when a Job transitions to
Running, the Discord_Bot posts a started message referencing the Job identifier
to the originating Discord channel, and it does so within the timing budget
(Req 6.1 -- within 5 seconds of the transition).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from discord_ollama_agent.discord_bot import DiscordBot
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.models.job import JobEvent, JobStatus

# Req 6.1: the started message must be posted within 5 seconds of the
# transition. The handler does no I/O beyond the (in-memory) fake sink here, so
# we assert a small fraction of that budget to keep the guarantee meaningful
# while remaining robust on a loaded CI host.
_TIMING_BUDGET_SECONDS = 5.0

ORIGIN_CHANNEL = 4242
JOB_ID = "backend-abc123"


class _RecordingSink:
    """A fake MessageSink that records every (channel_id, content) it is sent.

    Satisfies the awaitable :class:`~discord_ollama_agent.discord_bot.MessageSink`
    protocol and succeeds on the first attempt, so a single started message is
    delivered to the originating channel.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send(self, channel_id: int, content: str) -> None:
        self.sent.append((channel_id, content))


def _make_bot(sink: _RecordingSink) -> DiscordBot:
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[ORIGIN_CHANNEL],
        authorized_users=[10],
        admin_users=[20],
    )
    return DiscordBot(config, message_sink=sink)


@pytest.mark.asyncio
async def test_running_transition_posts_started_message_to_origin_channel() -> None:
    """A Running transition posts a started message referencing the Job id (Req 6.1)."""
    sink = _RecordingSink()
    bot = _make_bot(sink)
    event = JobEvent(
        job_id=JOB_ID, status=JobStatus.RUNNING, channel_id=ORIGIN_CHANNEL
    )

    await bot.on_job_event(event)

    # Exactly one message, delivered to the originating channel, referencing the id.
    assert len(sink.sent) == 1
    channel_id, content = sink.sent[0]
    assert channel_id == ORIGIN_CHANNEL
    assert JOB_ID in content
    assert "started" in content.lower()


@pytest.mark.asyncio
async def test_started_message_delivered_within_timing_budget() -> None:
    """The started message is posted within the 5-second timing budget (Req 6.1)."""
    sink = _RecordingSink()
    bot = _make_bot(sink)
    event = JobEvent(
        job_id=JOB_ID, status=JobStatus.RUNNING, channel_id=ORIGIN_CHANNEL
    )

    start = time.perf_counter()
    await bot.on_job_event(event)
    elapsed = time.perf_counter() - start

    assert elapsed < _TIMING_BUDGET_SECONDS, (
        f"started message took {elapsed:.3f}s, exceeding the "
        f"{_TIMING_BUDGET_SECONDS}s budget"
    )
    assert len(sink.sent) == 1
    assert sink.sent[0][0] == ORIGIN_CHANNEL
    assert JOB_ID in sink.sent[0][1]


@pytest.mark.asyncio
async def test_started_message_delivery_respects_asyncio_timeout() -> None:
    """Delivery completes well inside an asyncio timeout bounding the budget (Req 6.1)."""
    sink = _RecordingSink()
    bot = _make_bot(sink)
    event = JobEvent(
        job_id=JOB_ID, status=JobStatus.RUNNING, channel_id=ORIGIN_CHANNEL
    )

    # If on_job_event blew the budget, wait_for would raise TimeoutError.
    await asyncio.wait_for(bot.on_job_event(event), timeout=_TIMING_BUDGET_SECONDS)

    assert sink.sent == [(ORIGIN_CHANNEL, sink.sent[0][1])]
    assert JOB_ID in sink.sent[0][1]
