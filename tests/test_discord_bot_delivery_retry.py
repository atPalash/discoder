"""Property-based test for bounded status-delivery retries (task 12.8).

Covers Property 22: Status delivery retries are bounded.

*For any* sequence of delivery failures when posting a Job status message, the
Discord_Bot makes one initial attempt plus at most
:data:`~discord_ollama_agent.discord_bot._MAX_DELIVERY_RETRIES` retries -- four
attempts in all. If a send eventually succeeds within that bound, no further
attempts are made; if every attempt fails, no more than four sends occur, a
delivery-failure reason is recorded in the operator log, and the loop stops
(Req 6.7).

The bot is exercised through its real :meth:`DiscordBot.on_job_event` (and the
:meth:`DiscordBot._deliver` retry loop it drives) wired to a fake MessageSink
that raises on its first ``failures`` send attempts and counts every call. The
status to render is constrained to those that produce a Discord message
(Running/Succeeded/Failed) so a delivery is actually attempted.

**Validates: Requirements 6.7**
"""

from __future__ import annotations

import logging

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from discord_ollama_agent.discord_bot import (
    _MAX_DELIVERY_RETRIES,
    DiscordBot,
)
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.models.job import JobEvent, JobStatus

# The total attempt budget: one initial attempt plus the bounded retries.
_MAX_ATTEMPTS = _MAX_DELIVERY_RETRIES + 1

# Statuses that render a Discord status message, so delivery is attempted
# (Queued/Cancelled render nothing and are excluded here).
_DELIVERABLE_STATUSES = [
    JobStatus.RUNNING,
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
]


class _FlakyMessageSink:
    """A MessageSink that fails its first ``failures`` sends, then succeeds.

    Every call increments :attr:`attempts`; the first ``failures`` calls raise a
    delivery error and any subsequent call returns normally. With
    ``failures`` set at or above the attempt budget, every attempt the bot makes
    raises, which is where the upper bound and the give-up logging are observable.
    """

    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.attempts = 0

    async def send(self, channel_id: int, content: str) -> None:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise RuntimeError(f"simulated delivery failure #{self.attempts}")


def _make_bot(sink: _FlakyMessageSink) -> DiscordBot:
    """A minimal valid DiscordBot wired to ``sink`` for delivery (Req 13.5)."""
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[1],
    )
    return DiscordBot(config, message_sink=sink)


# Feature: discord-ollama-coding-agent, Property 22
# Property 22: Status delivery retries are bounded.
# Validates: Requirements 6.7
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    # 0..(budget + 2) failures: covers immediate success, success after some
    # retries, success on the very last allowed attempt, and total failure
    # (more failures than the bot will ever attempt).
    failures=st.integers(min_value=0, max_value=_MAX_ATTEMPTS + 2),
    status=st.sampled_from(_DELIVERABLE_STATUSES),
    channel_id=st.integers(min_value=1, max_value=10_000),
    job_suffix=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8
    ),
)
@pytest.mark.asyncio
async def test_status_delivery_retries_are_bounded(
    failures: int,
    status: JobStatus,
    channel_id: int,
    job_suffix: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Delivery is retried a bounded number of times then gives up with a log.

    For any number of leading send failures the bot must:
    - make at most ``1 + _MAX_DELIVERY_RETRIES`` (four) attempts, never more;
    - stop at the first success when one occurs within the bound, so the total
      attempts equal ``min(failures + 1, 1 + _MAX_DELIVERY_RETRIES)``;
    - when every attempt fails, make exactly four sends, attempt no more, and
      record a delivery-failure reason in the operator log (Req 6.7).
    """
    sink = _FlakyMessageSink(failures)
    bot = _make_bot(sink)
    event = JobEvent(
        job_id=f"acme-{job_suffix}",
        status=status,
        channel_id=channel_id,
        branch_name="feature/branch",
        failure_reason="boom",
    )

    # caplog is function-scoped and not reset between Hypothesis examples, so
    # clear prior records to keep each example's log assertions independent.
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="discord_ollama_agent.discord_bot"):
        await bot.on_job_event(event)

    delivery_failed = failures >= _MAX_ATTEMPTS
    expected_attempts = min(failures + 1, _MAX_ATTEMPTS)

    # --- The attempt bound: never more than 1 initial + 3 retries. ---
    assert sink.attempts <= _MAX_ATTEMPTS
    # Attempts stop at the first success, or at the budget when all fail.
    assert sink.attempts == expected_attempts

    delivery_failure_logged = any(
        record.levelno >= logging.ERROR and "Delivery failure" in record.getMessage()
        for record in caplog.records
    )
    if delivery_failed:
        # Every attempt failed: the budget was exhausted and the give-up reason
        # was recorded for the operator; no further attempts were made.
        assert sink.attempts == _MAX_ATTEMPTS
        assert delivery_failure_logged
        # The logged reason references the offending job (Req 6.7).
        assert any(event.job_id in record.getMessage() for record in caplog.records)
    else:
        # A send succeeded within the bound: no delivery-failure was logged.
        assert not delivery_failure_logged
