"""Property-based test for whitespace-only required-input rejection (task 12.4).

Covers Property 2: Whitespace-only required inputs are rejected.

*For any* string composed entirely of Unicode whitespace supplied as a
``/build`` idea or a ``/revise`` feedback argument, the System creates no
Job/revision and replies requesting non-empty input.

**Validates: Requirements 1.5, 11.8**

These tests drive the real :class:`DiscordBot` command handlers
(:meth:`~discord_ollama_agent.discord_bot.DiscordBot.on_build` and
:meth:`~discord_ollama_agent.discord_bot.DiscordBot.on_revise`) wired to a real
:class:`~discord_ollama_agent.job_manager.JobManager` and a small fake
Target_Registry. The gate is always satisfied (allowed channel + authorized
user) so the only thing under test is the non-empty-input guard: the handler
must short-circuit before any Job/revision is minted and return the
corresponding request-non-empty-input reply.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.discord_bot import (
    DiscordBot,
    _EMPTY_FEEDBACK_MESSAGE,
    _EMPTY_IDEA_MESSAGE,
)
from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget

ALLOWED_CHANNEL = 1000
USER = 10
ADMIN = 20
TARGET_NAME = "svc"


@dataclass(frozen=True)
class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    channel_id: int
    user_id: int


class _FakeRegistry:
    """A tiny Target_Registry stand-in resolving a single usable target.

    Only :meth:`get` is exercised by ``/build``; it resolves the configured
    target name and reports every other name as not registered. The handler
    rejects a whitespace-only idea *before* it ever resolves a target, so this
    fake exists only to prove the guard fires ahead of any registry/manager use.
    """

    def __init__(self, target: RegisteredTarget) -> None:
        self._target = target

    def get(self, name: str) -> RegisteredTarget | None:
        return self._target if name == self._target.name else None


def _make_target(name: str = TARGET_NAME) -> RegisteredTarget:
    """A minimal valid target used to resolve ``/build`` requests."""
    return RegisteredTarget(
        name=name,
        directory_path="workspace/t",
        repo_remote="git@example.com:org/repo.git",
        credentials_ref="GIT_TOKEN",
    )


def _make_bot() -> tuple[DiscordBot, JobManager]:
    """Build a DiscordBot wired to a real JobManager and a fake registry."""
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[ALLOWED_CHANNEL],
        authorized_users=[USER],
        admin_users=[ADMIN],
        default_target=TARGET_NAME,
    )
    manager = JobManager()
    registry = _FakeRegistry(_make_target())
    bot = DiscordBot(config, job_manager=manager, target_registry=registry)
    return bot, manager


def _register_succeeded_base_job(manager: JobManager) -> Job:
    """Place a revisable Succeeded base Job (with a branch) into the manager.

    This is the only state from which ``/revise`` could otherwise create a
    revision, so any whitespace-feedback rejection cannot be attributed to an
    ineligible base Job.
    """
    base = Job(
        id=f"{TARGET_NAME}-base",
        target_name=TARGET_NAME,
        idea="add a healthcheck endpoint",
        status=JobStatus.SUCCEEDED,
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=ALLOWED_CHANNEL,
        user_id=USER,
        branch_name="agent/svc-base",
    )
    manager._jobs[base.id] = base
    return base


# Whitespace-only strings: ordinary spaces/tabs/newlines plus assorted Unicode
# whitespace (NBSP, EM SPACE, line/paragraph separators, ideographic space),
# including the empty string. Every generated value satisfies ``not s.strip()``.
_WHITESPACE_ALPHABET = " \t\n\r\f\v\u00a0\u2003\u2028\u2029\u3000\u200a\u2009"
_whitespace_only = st.text(alphabet=_WHITESPACE_ALPHABET, min_size=0, max_size=12)

# ``/build`` accepts an optional target; exercise both the omitted (None) and
# explicitly supplied target forms to show the idea guard fires regardless.
_targets = st.one_of(st.none(), st.just(TARGET_NAME))


# Feature: discord-ollama-coding-agent, Property 2
# Property 2: Whitespace-only required inputs are rejected.
# Validates: Requirements 1.5, 11.8
@settings(max_examples=20, deadline=None)
@given(idea=_whitespace_only, target=_targets)
def test_build_rejects_whitespace_only_idea(idea: str, target: str | None) -> None:
    """A whitespace-only ``/build`` idea creates no Job and asks for non-empty input."""
    # Precondition the generator guarantees: the idea is entirely whitespace.
    assert idea.strip() == ""

    bot, manager = _make_bot()
    jobs_before = len(manager._jobs)
    queue_before = len(manager._queue)

    reply = asyncio.run(
        bot.on_build(_Interaction(ALLOWED_CHANNEL, USER), target, idea)
    )

    # No Job was created or enqueued (Req 1.5).
    assert len(manager._jobs) == jobs_before
    assert len(manager._queue) == queue_before

    # The reply requests non-empty input and is non-empty itself (Req 1.5).
    assert reply == _EMPTY_IDEA_MESSAGE
    assert reply.strip() != ""


# Feature: discord-ollama-coding-agent, Property 2
# Property 2: Whitespace-only required inputs are rejected.
# Validates: Requirements 1.5, 11.8
@settings(max_examples=20, deadline=None)
@given(feedback=_whitespace_only)
def test_revise_rejects_whitespace_only_feedback(feedback: str) -> None:
    """A whitespace-only ``/revise`` feedback creates no revision and asks for
    non-empty input, even against a revisable Succeeded base Job (Req 11.8)."""
    # Precondition the generator guarantees: the feedback is entirely whitespace.
    assert feedback.strip() == ""

    bot, manager = _make_bot()
    base = _register_succeeded_base_job(manager)
    jobs_before = len(manager._jobs)
    queue_before = len(manager._queue)

    reply = asyncio.run(
        bot.on_revise(_Interaction(ALLOWED_CHANNEL, USER), base.id, feedback)
    )

    # No revision Job was created or enqueued (Req 11.8).
    assert len(manager._jobs) == jobs_before
    assert len(manager._queue) == queue_before

    # The reply requests non-empty feedback and is non-empty itself (Req 11.8).
    assert reply == _EMPTY_FEEDBACK_MESSAGE
    assert reply.strip() != ""
