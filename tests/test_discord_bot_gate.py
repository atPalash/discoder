"""Unit tests for the Discord_Bot command gate (task 12.1).

These example-based tests exercise :meth:`DiscordBot._gate`, the centralized
gating conjunction every command must satisfy before being acted upon: the
originating channel is among the configured ``Allowed_Channels`` AND the
invoking user is authorized for that command (the admin allowlist for
``/addtarget``, the user allowlist otherwise). They cover the truth table of the
conjunction plus the precedence of the channel check over the user check
(Req 1.4, 6.6, 8.1, 9.3, 10.2, 11.5, 13.1, 13.2, 13.3).

The exhaustive over-all-inputs assertion lives in the Property 1 test (task
12.2); these tests pin specific representative examples and edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass

from discord_ollama_agent.discord_bot import DiscordBot, GateDecision
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)


@dataclass(frozen=True)
class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    channel_id: int
    user_id: int


ALLOWED_CHANNEL = 1000
DENIED_CHANNEL = 2000
USER = 10
ADMIN = 20
STRANGER = 30


def _make_bot() -> DiscordBot:
    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[ALLOWED_CHANNEL],
        authorized_users=[USER],
        admin_users=[ADMIN],
    )
    return DiscordBot(config)


def test_allowed_channel_and_authorized_user_passes() -> None:
    """An authorized user in an allowed channel is allowed (Req 13.2)."""
    result = _make_bot()._gate(_Interaction(ALLOWED_CHANNEL, USER), admin=False)
    assert result.allowed is True
    assert result.decision is GateDecision.ALLOWED
    assert result.message is None


def test_denied_channel_blocks_even_authorized_user() -> None:
    """A disallowed channel refuses even an authorized user (Req 13.1, 13.3)."""
    result = _make_bot()._gate(_Interaction(DENIED_CHANNEL, USER), admin=False)
    assert result.allowed is False
    assert result.decision is GateDecision.CHANNEL_NOT_PERMITTED
    assert result.message


def test_allowed_channel_unauthorized_user_blocked() -> None:
    """An allowed channel but unlisted user is refused (Req 1.4, 6.6, 8.1, 9.3, 11.5)."""
    result = _make_bot()._gate(_Interaction(ALLOWED_CHANNEL, STRANGER), admin=False)
    assert result.allowed is False
    assert result.decision is GateDecision.USER_NOT_AUTHORIZED
    assert result.message


def test_channel_check_takes_precedence_over_user_check() -> None:
    """A stranger in a denied channel reports the channel denial first (Req 13.1, 13.3)."""
    result = _make_bot()._gate(_Interaction(DENIED_CHANNEL, STRANGER), admin=False)
    assert result.decision is GateDecision.CHANNEL_NOT_PERMITTED


def test_addtarget_uses_admin_allowlist_not_user_allowlist() -> None:
    """An ordinary authorized user cannot run the admin-gated command (Req 10.2)."""
    result = _make_bot()._gate(_Interaction(ALLOWED_CHANNEL, USER), admin=True)
    assert result.allowed is False
    assert result.decision is GateDecision.USER_NOT_AUTHORIZED


def test_addtarget_allows_admin_user() -> None:
    """An admin user in an allowed channel may run the admin-gated command (Req 10.2)."""
    result = _make_bot()._gate(_Interaction(ALLOWED_CHANNEL, ADMIN), admin=True)
    assert result.allowed is True
    assert result.decision is GateDecision.ALLOWED


def test_admin_user_not_implicitly_authorized_for_ordinary_commands() -> None:
    """Admin allowlist is independent of the user allowlist (Req 8.1, definitions)."""
    result = _make_bot()._gate(_Interaction(ALLOWED_CHANNEL, ADMIN), admin=False)
    assert result.allowed is False
    assert result.decision is GateDecision.USER_NOT_AUTHORIZED
