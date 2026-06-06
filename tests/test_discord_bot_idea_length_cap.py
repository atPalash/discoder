"""Property-based test for the /build idea length cap (task 12.5).

Covers Property 3: the idea length cap is enforced.

*For any* configured maximum idea length and *for any* idea string, the
Discord_Bot's ``/build`` handler creates a Queued Job **if and only if** the
idea's length measured in Unicode characters (codepoints) does not exceed the
configured maximum; otherwise it replies with a too-long message and creates no
Job (Req 1.8).

The test drives the real :meth:`DiscordBot.on_build` against a real
:class:`JobManager` and a real :class:`TargetRegistry` holding a single usable
target, so the command gate passes and the target resolves -- leaving the length
cap as the only decision under test. Ideas are generated from non-whitespace
single-codepoint characters across lengths below, at, and above the configured
maximum, so the empty-idea check (Property 2) never fires and ``len(idea)`` is
exactly the generated codepoint count.

**Validates: Requirements 1.8**
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.discord_bot import DiscordBot
from discord_ollama_agent.job_manager import JobManager
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.target_registry import TargetRegistry

ALLOWED_CHANNEL = 1000
AUTHORIZED_USER = 10
TARGET_NAME = "demo"

# Single-codepoint, guaranteed-non-whitespace characters: excluding the control
# (Cc), surrogate (Cs), and separator (Zs/Zl/Zp) categories removes every
# Unicode character that ``str.strip()`` treats as whitespace, so a string built
# from these always survives the non-empty-idea check and only the length cap
# decides. Each drawn character is exactly one codepoint, so ``len`` of the
# joined string equals the number of characters drawn.
_NON_WHITESPACE_CHAR = st.characters(
    blacklist_categories=("Cc", "Cs", "Zs", "Zl", "Zp"),
)


@st.composite
def _max_and_idea(draw: st.DrawFn) -> tuple[int, str]:
    """Draw a configured maximum and an idea spanning below/at/above that max.

    The idea length ranges over ``[1, max_len + 10]`` so generated cases land
    below, exactly at, and above the configured maximum; the lower bound of 1
    keeps the idea non-empty so the length cap (not the empty-idea check) is the
    deciding rule.
    """
    max_len = draw(st.integers(min_value=1, max_value=40))
    length = draw(st.integers(min_value=1, max_value=max_len + 10))
    chars = draw(
        st.lists(_NON_WHITESPACE_CHAR, min_size=length, max_size=length)
    )
    return max_len, "".join(chars)


def _build_registry(root: Path) -> TargetRegistry:
    """Create a registry on disk with one usable target and load it.

    The target's directory resolves inside the workspace and it carries a
    non-empty remote, so it lands in the usable set and ``/build`` resolves it.
    """
    workspace = root / "workspace"
    target_dir = workspace / TARGET_NAME
    target_dir.mkdir(parents=True)
    registry_path = root / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": TARGET_NAME,
                        "directory_path": str(target_dir),
                        "repo_remote": "https://example.invalid/repo.git",
                        "credentials_ref": "GIT_TOKEN_ENV",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = TargetRegistry()
    registry.load(registry_path, workspace)
    return registry


class _Interaction:
    """A minimal GateContext stand-in exposing channel_id and user_id."""

    def __init__(self, channel_id: int, user_id: int) -> None:
        self.channel_id = channel_id
        self.user_id = user_id


# Feature: discord-ollama-coding-agent, Property 3
# Property 3: Idea length cap is enforced.
# Validates: Requirements 1.8
@settings(max_examples=20, deadline=None)
@given(case=_max_and_idea())
async def test_build_creates_job_iff_idea_within_length_cap(
    case: tuple[int, str],
) -> None:
    """A Job is created iff the idea's Unicode length is within the cap (Req 1.8).

    With the gate passing and the target resolving, the only deciding rule is
    the configured maximum idea length: when ``len(idea) <= max`` exactly one
    Queued Job is created and the reply confirms it; when ``len(idea) > max`` no
    Job is created and the reply states the idea exceeds the maximum.
    """
    max_len, idea = case
    # Each character is one codepoint, so this is the Unicode-length oracle.
    within_cap = len(idea) <= max_len

    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        registry = _build_registry(root)
        config = SystemConfig(
            ollama=OllamaConfig(
                endpoint_url="http://localhost:11434", model="llama3"
            ),
            workspace_dir=str(root / "workspace"),
            concurrency_limit=4,
            context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
            allowed_channels=[ALLOWED_CHANNEL],
            authorized_users=[AUTHORIZED_USER],
            max_idea_length=max_len,
        )
        manager = JobManager(concurrency_limit=4)
        bot = DiscordBot(config, job_manager=manager, target_registry=registry)

        reply = await bot.on_build(
            _Interaction(ALLOWED_CHANNEL, AUTHORIZED_USER),
            target=TARGET_NAME,
            idea=idea,
        )

        if within_cap:
            # A single Queued Job is created and the reply references it.
            assert len(manager._jobs) == 1
            (job,) = manager._jobs.values()
            assert job.target_name == TARGET_NAME
            assert reply == (
                f"Created job `{job.id}` (Queued) for target {TARGET_NAME!r}."
            )
        else:
            # No Job is created and the reply reports the too-long idea.
            assert manager._jobs == {}
            assert reply == (
                f"The idea exceeds the configured maximum length of {max_len} "
                "characters."
            )
