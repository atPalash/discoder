"""Property-based test for the Discord_Bot command gate (task 12.2).

This module exercises the universal gating conjunction enforced by
:meth:`DiscordBot._gate` across the whole input space of channels, users, and
the admin flag (Property 1). The companion example-based suite lives in
``test_discord_bot_gate.py``; this test asserts the *iff* over all generated
inputs rather than pinning specific cases.
"""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

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


def _make_bot(
    allowed_channels: list[int],
    authorized_users: list[int],
    admin_users: list[int],
) -> DiscordBot:
    """Build a DiscordBot whose config carries the given allowlists.

    ``allowed_channels`` must be non-empty (SystemConfig enforces Req 13.5); the
    caller guarantees this from the generated membership pools below.
    """

    config = SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir="/tmp/agent-workspace",
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=allowed_channels,
        authorized_users=authorized_users,
        admin_users=admin_users,
    )
    return DiscordBot(config)


# Disjoint identifier pools so that the generated channel/user can be either a
# member or a non-member of each allowlist, covering allowed/denied membership.
_CHANNEL_POOL = [1000, 1001, 1002, 1003]
_USER_POOL = [10, 11, 12, 13, 14, 15]


# Feature: discord-ollama-coding-agent, Property 1
# Property 1: Command gating requires both channel and user authorization.
# Validates: Requirements 1.4, 6.6, 8.1, 9.3, 10.2, 11.5, 13.1, 13.2, 13.3
@given(
    # A non-empty subset of channels is allowed (Req 13.5 requires at least one).
    allowed_channels=st.lists(
        st.sampled_from(_CHANNEL_POOL), min_size=1, max_size=len(_CHANNEL_POOL), unique=True
    ),
    authorized_users=st.lists(
        st.sampled_from(_USER_POOL), max_size=len(_USER_POOL), unique=True
    ),
    admin_users=st.lists(
        st.sampled_from(_USER_POOL), max_size=len(_USER_POOL), unique=True
    ),
    channel_id=st.sampled_from(_CHANNEL_POOL),
    user_id=st.sampled_from(_USER_POOL),
    admin=st.booleans(),
)
def test_gate_allows_iff_channel_allowed_and_user_authorized(
    allowed_channels: list[int],
    authorized_users: list[int],
    admin_users: list[int],
    channel_id: int,
    user_id: int,
    admin: bool,
) -> None:
    """The gate is acted upon iff the channel is allowed AND the user is
    authorized for that command; otherwise no allowance and a denial reply.

    The applicable allowlist is the admin allowlist for ``/addtarget``
    (``admin=True``) and the user allowlist for every other command
    (``admin=False``) (Req 1.4, 6.6, 8.1, 9.3, 10.2, 11.5, 13.1, 13.2, 13.3).
    """

    bot = _make_bot(allowed_channels, authorized_users, admin_users)
    result = bot._gate(_Interaction(channel_id, user_id), admin=admin)

    channel_ok = channel_id in allowed_channels
    applicable_allowlist = admin_users if admin else authorized_users
    user_ok = user_id in applicable_allowlist

    expected_allowed = channel_ok and user_ok

    # The conjunction: allowed iff both checks pass.
    assert result.allowed is expected_allowed

    if expected_allowed:
        # On allowance: the ALLOWED decision and no denial reply (Req 13.2).
        assert result.decision is GateDecision.ALLOWED
        assert result.message is None
    else:
        # On denial: never ALLOWED, and a ready-to-send reply is produced
        # (no state change is the caller's contract; the gate signals refusal).
        assert result.decision is not GateDecision.ALLOWED
        assert result.message

        # The channel check takes precedence over the user check (Req 13.1, 13.3):
        # a disallowed channel always reports CHANNEL_NOT_PERMITTED regardless of
        # the user's authorization; only an allowed channel can surface a
        # USER_NOT_AUTHORIZED denial.
        if not channel_ok:
            assert result.decision is GateDecision.CHANNEL_NOT_PERMITTED
        else:
            assert result.decision is GateDecision.USER_NOT_AUTHORIZED
