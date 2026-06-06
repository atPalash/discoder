"""Property-based test for empty-generation handling (task 9.6).

Covers Property 28: Zero-file generations fail as empty-generation.

*For any* completion that parses into a Generation_Result containing zero file
entries, :meth:`CodingAgent.run` transitions the Job to Failed with the
``empty-generation`` reason and performs **no** write into the sandbox; any
completion that parses into at least one file entry instead proceeds to the
write step (Req 3.12).

The agent is exercised end-to-end through :meth:`CodingAgent.run` with in-memory
fakes for its collaborators: a fake Ollama client returns a single, deterministic
completion built from the generated file set; a fake sandbox records every
write so "did the write step run?" is observed directly; a fake Git manager
stands in for sync/commit/push; and a fake registry resolves the Job's target.

The oracle ("does the generated result carry zero files?") is derived purely
from the generated input (test knowledge), never by re-running the agent's own
logic, so the test independently pins down the branch.

**Validates: Requirements 3.12**
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.coding_agent import (
    EMPTY_GENERATION_REASON,
    CodingAgent,
)
from discord_ollama_agent.execution_sandbox import CancellationToken, CommandResult
from discord_ollama_agent.git_manager import PushResult, PushStatus, SyncResult
from discord_ollama_agent.models.generation import FileEntry, GenerationResult
from discord_ollama_agent.models.job import Job, JobEvent, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.ollama_client import RawCompletion

# A single legal path component: lowercase letters/digits only, so a drawn
# segment is always a valid file name.
_SAFE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=12,
)

# File contents restricted to characters that round-trip cleanly through JSON.
_TEXT_CONTENT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=64,
)

# A single file entry the model "generated": a name plus its content.
_FILE_SPEC = st.fixed_dictionaries({"name": _SAFE_NAME, "content": _TEXT_CONTENT})


class _FakeOllamaClient:
    """Returns a fixed completion string for every generation call."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.generate_calls = 0

    async def generate(self, request) -> RawCompletion:
        self.generate_calls += 1
        return RawCompletion(content=self._content)


class _FakeSandbox:
    """Records every write so the write step is observable; no real filesystem."""

    def __init__(self, working_dir: Path) -> None:
        self._working_dir = working_dir
        self.writes: list[tuple[str, str]] = []
        self.run_command_calls = 0

    def init_working_dir(self, job: Job, source: Path) -> Path:
        return self._working_dir

    def write_file(self, relative_path: str, content: str) -> None:
        self.writes.append((relative_path, content))

    async def run_command(self, command, cancel, timeout_s) -> CommandResult:
        self.run_command_calls += 1
        return CommandResult(exit_code=0, output="")


class _FakeGitManager:
    """Stands in for sync/commit/push; sync always succeeds, push always pushes."""

    def __init__(self) -> None:
        self.sync_calls = 0
        self.commit_calls = 0

    async def sync_default_branch(self, target, working_copy) -> SyncResult:
        self.sync_calls += 1
        return SyncResult(ok=True)

    async def commit_and_push(self, job, target, working_copy) -> PushResult:
        self.commit_calls += 1
        return PushResult(status=PushStatus.PUSHED, branch_name="agent/branch")

    async def commit_and_push_revision(self, job, target, working_copy) -> PushResult:
        self.commit_calls += 1
        return PushResult(status=PushStatus.PUSHED, branch_name="agent/branch")


class _FakeRegistry:
    """Resolves the single target the Job references."""

    def __init__(self, target: RegisteredTarget) -> None:
        self._target = target

    def get(self, name: str) -> RegisteredTarget | None:
        if name == self._target.name:
            return self._target
        return None


def _make_target() -> RegisteredTarget:
    """A target with no Build_Command, so a successful write proceeds to commit."""
    return RegisteredTarget(
        name="acme",
        directory_path=Path("/workspace/acme"),
        repo_remote="https://example.invalid/repo.git",
        build_command=None,
        credentials_ref="GIT_TOKEN_ENV",
    )


def _make_job() -> Job:
    return Job(
        id="acme-job1",
        target_name="acme",
        idea="build something",
        status=JobStatus.QUEUED,
        submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        channel_id=1,
        user_id=2,
    )


# Feature: discord-ollama-coding-agent, Property 28
# Property 28: Zero-file generations fail as empty-generation
# Validates: Requirements 3.12
@settings(max_examples=20, deadline=None)
@given(specs=st.lists(_FILE_SPEC, min_size=0, max_size=5))
@pytest.mark.asyncio
async def test_zero_file_generation_fails_as_empty_generation(specs: list[dict]):
    """A zero-file result fails empty-generation with no write; >=1 file writes.

    Builds a completion from ``specs`` (the file set the model "returned") and
    runs the full agent lifecycle. The oracle is ``len(specs) == 0``: when no
    files are present the Job must end Failed/empty-generation with no sandbox
    write; otherwise the agent must reach the write step (Req 3.12).
    """
    # Distinct paths so every spec yields exactly one file entry.
    entries = [
        FileEntry(path=f"f{index}/{spec['name']}", content=spec["content"])
        for index, spec in enumerate(specs)
    ]
    completion = GenerationResult(files=entries).model_dump_json()

    target = _make_target()
    job = _make_job()

    ollama = _FakeOllamaClient(completion)
    sandbox = _FakeSandbox(Path("/workspace/acme/work"))
    git = _FakeGitManager()

    events: list[JobEvent] = []

    async def sink(event: JobEvent) -> None:
        events.append(event)

    agent = CodingAgent(
        sandbox=sandbox,
        ollama_client=ollama,
        git_manager=git,
        config=None,
        registry=_FakeRegistry(target),
        event_sink=sink,
    )

    await agent.run(job, CancellationToken())

    # The completion is always well-formed JSON, so generation runs exactly once
    # and parsing always succeeds; the branch turns solely on the file count.
    assert ollama.generate_calls == 1

    if not entries:
        # Zero file entries -> terminal empty-generation failure, and the write
        # step is never reached (Req 3.12).
        assert job.status == JobStatus.FAILED
        assert job.failure_reason == EMPTY_GENERATION_REASON
        assert sandbox.writes == []
        assert git.commit_calls == 0
        # A Failed event was emitted carrying the empty-generation reason.
        assert events[-1].status == JobStatus.FAILED
        assert events[-1].failure_reason == EMPTY_GENERATION_REASON
    else:
        # At least one file entry -> the agent proceeds to the write step,
        # writing exactly the generated entries (Req 3.12).
        assert job.failure_reason != EMPTY_GENERATION_REASON
        assert sandbox.writes == [(entry.path, entry.content) for entry in entries]
        # Having written, it continues past the write step to commit/push.
        assert git.commit_calls == 1
        assert job.status == JobStatus.SUCCEEDED
