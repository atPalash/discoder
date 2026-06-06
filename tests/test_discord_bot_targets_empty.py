"""Unit test for ``/targets`` against an empty registry (task 12.9).

Exercises :meth:`DiscordBot.on_targets` for the empty-registry case: when the
gate passes (an authorized user invoking from an allowed channel) but no usable
Registered_Targets exist, the reply indicates that no targets are configured and
discloses no names (Req 9.2).

The exhaustive over-all-inputs assertions live in the gate property test (task
12.2); this pins the specific empty-registry example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from discord_ollama_agent.discord_bot import DiscordBot
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.target_registry import TargetRegistry


@dataclass(frozen=True)
class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    channel_id: int
    user_id: int


ALLOWED_CHANNEL = 1000
USER = 10
ADMIN = 20


def _make_config(workspace_dir: str) -> SystemConfig:
    return SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="llama3"),
        workspace_dir=workspace_dir,
        concurrency_limit=4,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[ALLOWED_CHANNEL],
        authorized_users=[USER],
        admin_users=[ADMIN],
    )


async def test_targets_empty_registry_replies_no_targets_configured() -> None:
    """``/targets`` with no Registered_Targets reports none are configured (Req 9.2)."""
    with TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "agent-workspace"
        workspace.mkdir()

        # A fresh registry loaded against a missing file has no usable targets.
        registry = TargetRegistry()
        registry.load(workspace / "registry.json", workspace)
        assert registry.names() == []

        bot = DiscordBot(_make_config(str(workspace)), target_registry=registry)

        reply = await bot.on_targets(_Interaction(ALLOWED_CHANNEL, USER))

    assert "no targets" in reply.lower()
