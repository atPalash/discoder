"""Property-based test for not-found replies (task 12.7).

Covers Property 21: Not-found replies reference the queried identifier.

*For any* ``/status``, ``/cancel``, or ``/revise`` invocation referencing a Job
identifier that is not present in the Job table, the Discord_Bot replies with a
not-found message that includes the queried identifier and makes no state change
(Req 6.5, 8.2, 11.6).

The handlers are exercised against a real, empty :class:`JobManager` so every
generated id is genuinely unknown. An authorized invocation (allowed channel,
authorized user) is used so the gate passes and the not-found branch -- rather
than an authorization denial -- is what produces the reply. After each call the
manager is asserted to still hold no Jobs and an empty queue, which is what makes
the "no state changes" claim observable.

**Validates: Requirements 6.5, 8.2, 11.6**
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.discord_bot import DiscordBot, _not_found_message
from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)


_ALLOWED_CHANNEL = 1000
_AUTHORIZED_USER = 10


@dataclass(frozen=True)
class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    channel_id: int
    user_id: int


def _make_bot(manager: JobManager) -> DiscordBot:
    """Build a DiscordBot wired to ``manager`` with a permissive gate config.

    The allowed channel and authorized user below match the interaction used in
    the test, so the gate passes and each handler reaches its not-found branch.
    """
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[_ALLOWED_CHANNEL],
        authorized_users=[_AUTHORIZED_USER],
        admin_users=[_AUTHORIZED_USER],
    )
    return DiscordBot(config, job_manager=manager)


def _assert_no_state_change(manager: JobManager) -> None:
    """Assert no Job/revision state was created by a not-found invocation."""
    assert manager._jobs == {}
    assert len(manager._queue) == 0
    assert manager.running_count == 0


# Feature: discord-ollama-coding-agent, Property 21
# Property 21: Not-found replies reference the queried identifier.
# Validates: Requirements 6.5, 8.2, 11.6
@settings(max_examples=20, deadline=None)
@given(
    job_id=st.text(min_size=1, max_size=80),
    feedback=st.text(min_size=1, max_size=80).filter(lambda s: s.strip()),
)
@pytest.mark.asyncio
async def test_not_found_replies_reference_queried_identifier(
    job_id: str,
    feedback: str,
) -> None:
    """/status, /cancel, /revise on an unknown id reply not-found and mutate nothing."""
    expected = _not_found_message(job_id)
    itx = _Interaction(_ALLOWED_CHANNEL, _AUTHORIZED_USER)

    # /status against an unknown id (Req 6.5).
    manager = JobManager()
    bot = _make_bot(manager)
    status_reply = await bot.on_status(itx, job_id)
    assert status_reply == expected
    assert job_id in status_reply
    _assert_no_state_change(manager)

    # /cancel against an unknown id (Req 8.2).
    manager = JobManager()
    bot = _make_bot(manager)
    cancel_reply = await bot.on_cancel(itx, job_id)
    assert cancel_reply == expected
    assert job_id in cancel_reply
    _assert_no_state_change(manager)

    # /revise against an unknown id, with non-empty feedback so the empty-feedback
    # guard does not pre-empt the not-found branch (Req 11.6, 11.8).
    manager = JobManager()
    bot = _make_bot(manager)
    revise_reply = await bot.on_revise(itx, job_id, feedback)
    assert revise_reply == expected
    assert job_id in revise_reply
    _assert_no_state_change(manager)
