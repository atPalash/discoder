"""Example test for ``/build`` with an omitted target and no Default_Target (task 12.11).

Pins the Req 1.7 behaviour of :meth:`DiscordBot.on_build`: when the target name
argument is omitted **and** no Default_Target is configured, the command makes
no state change -- no Job is created -- and replies with a message indicating a
target must be specified. The gate is satisfied (authorized user in an allowed
channel) and the idea is a valid non-whitespace string, so the only reason the
command stops short is the missing target.

Validates: Requirements 1.7
"""

from __future__ import annotations

from dataclasses import dataclass

from discord_ollama_agent.discord_bot import DiscordBot, _NO_TARGET_MESSAGE
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)


ALLOWED_CHANNEL = 1000
USER = 10


@dataclass(frozen=True)
class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    channel_id: int
    user_id: int


class _RecordingJobManager:
    """A spy that records whether the bot attempted to create or queue a Job.

    The no-target path must short-circuit before any Job_Manager interaction, so
    these counters let the test assert that no Job was created (Req 1.7).
    """

    def __init__(self) -> None:
        self.create_job_calls = 0
        self.notify_calls = 0

    def create_job(self, *args: object, **kwargs: object) -> object:  # pragma: no cover - must not run
        self.create_job_calls += 1
        raise AssertionError("create_job must not be called when no target is resolvable")

    def notify(self) -> None:  # pragma: no cover - must not run
        self.notify_calls += 1


def _make_bot(manager: _RecordingJobManager) -> DiscordBot:
    """Build a DiscordBot whose config has no Default_Target (the Req 1.7 case)."""
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[ALLOWED_CHANNEL],
        authorized_users=[USER],
        # default_target is left unset (None): no Default_Target is configured.
    )
    assert config.default_target is None
    return DiscordBot(config, job_manager=manager)


async def test_omitted_target_without_default_creates_no_job_and_asks_for_target() -> None:
    """Omitted target + no Default_Target: no Job created, reply asks for a target (Req 1.7)."""
    manager = _RecordingJobManager()
    bot = _make_bot(manager)

    reply = await bot.on_build(
        _Interaction(ALLOWED_CHANNEL, USER), target=None, idea="add a healthcheck endpoint"
    )

    # No Job was created and the dispatcher was never notified (no state change).
    assert manager.create_job_calls == 0
    assert manager.notify_calls == 0
    # The reply indicates that a target must be specified.
    assert reply == _NO_TARGET_MESSAGE


async def test_blank_target_without_default_is_treated_as_omitted() -> None:
    """A whitespace-only target with no Default_Target is also rejected (Req 1.7).

    The handler treats a blank target name the same as an omitted one, so this
    likewise creates no Job and asks for a target.
    """
    manager = _RecordingJobManager()
    bot = _make_bot(manager)

    reply = await bot.on_build(
        _Interaction(ALLOWED_CHANNEL, USER), target="   ", idea="add a healthcheck endpoint"
    )

    assert manager.create_job_calls == 0
    assert manager.notify_calls == 0
    assert reply == _NO_TARGET_MESSAGE
