"""Property-based test for attempt-bounded retry with feedback (task 9.5).

Covers Property 14: Retry is attempt-bounded and feeds prior errors back.

*For any* sequence of failing attempts -- a malformed Generation_Result or a
non-zero build exit -- :meth:`CodingAgent.run` drives a bounded retry loop:
while the number of completed attempts is below ``Maximum_Build_Attempts`` a new
attempt begins carrying the prior error (the parse error or the captured build
output) as feedback; once completed attempts reach ``Maximum_Build_Attempts``
the Job transitions to Failed with the corresponding reason
(``invalid-output`` for malformed output, ``max-attempts-exhausted`` for a
non-zero build), and the total number of attempts never exceeds
``Maximum_Build_Attempts`` (Req 3.8, 3.9, 4.5, 4.6).

The agent is exercised through its real :meth:`run` loop wired to in-memory
fakes: a fake Ollama client that always returns a failing completion (malformed
text, or a valid result whose build then fails), a fake sandbox whose build
command always exits non-zero, a fake Git manager whose sync succeeds, and a
fake registry returning a single target. Because every attempt fails, the loop
is forced to run to its budget, which is exactly where the attempt bound and the
feedback chaining are observable.

**Validates: Requirements 3.8, 3.9, 4.5, 4.6**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.coding_agent import (
    INVALID_OUTPUT_REASON,
    MAX_ATTEMPTS_EXHAUSTED_REASON,
    CodingAgent,
)
from discord_ollama_agent.execution_sandbox import CancellationToken, CommandResult
from discord_ollama_agent.git_manager import SyncResult
from discord_ollama_agent.models.config import (
    ContextBudget,
    OllamaConfig,
    SystemConfig,
)
from discord_ollama_agent.models.generation import (
    FileEntry,
    GenerationRequest,
    GenerationResult,
)
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.ollama_client import RawCompletion

# Sentinel prefix the fake sandbox stamps on each build's captured output, so a
# retry request's ``prior_error`` can be matched against the exact prior build
# output it should carry back (Req 4.5).
_BUILD_OUTPUT_PREFIX = "BUILD_FAIL"

# A single legal target/file name component: lowercase letters and digits only.
_SAFE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=12
)


class _FakeOllamaClient:
    """Records every generation request and always returns a failing completion.

    In ``"malformed"`` mode every completion is unparseable text, so each attempt
    fails to parse into a Generation_Result (Req 3.7-3.9). In ``"build"`` mode
    every completion is a valid, non-empty Generation_Result, so parsing and
    writing succeed and the (always-failing) build step is what fails the attempt
    (Req 4.5, 4.6).
    """

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> RawCompletion:
        self.requests.append(request)
        if self._mode == "malformed":
            return RawCompletion(content="<<< this is not valid generation JSON >>>")
        result = GenerationResult(
            files=[FileEntry(path="generated.txt", content="placeholder")]
        )
        return RawCompletion(content=result.model_dump_json())


class _FakeSandbox:
    """In-memory sandbox: writes are no-ops and the build always exits non-zero.

    Each :meth:`run_command` returns a distinct, index-stamped output so the test
    can verify that the next attempt's request carries *that* build output back
    as its prior error (Req 4.5).
    """

    def __init__(self, working_dir: Path, exit_code: int) -> None:
        self._working_dir = working_dir
        self._exit_code = exit_code
        self.run_calls = 0
        self.written: list[tuple[str, str]] = []

    def init_working_dir(self, job: Job, source: Path) -> Path:
        return self._working_dir

    def write_file(self, relative_path: str, content: str) -> None:
        self.written.append((relative_path, content))

    async def run_command(
        self, command: str, cancel: CancellationToken, timeout_s: int
    ) -> CommandResult:
        index = self.run_calls
        self.run_calls += 1
        return CommandResult(
            exit_code=self._exit_code,
            output=f"{_BUILD_OUTPUT_PREFIX}::{index}",
        )


class _FakeGitManager:
    """Sync always succeeds; commit/push is never reached on a failing loop."""

    async def sync_default_branch(
        self, target: RegisteredTarget, working_copy: Path
    ) -> SyncResult:
        return SyncResult(ok=True)


class _FakeRegistry:
    """Resolves the single target the Job runs against."""

    def __init__(self, target: RegisteredTarget) -> None:
        self._target = target

    def get(self, name: str) -> RegisteredTarget | None:
        return self._target


def _make_config(max_build_attempts: int) -> SystemConfig:
    """A minimal valid SystemConfig with the chosen attempt bound (Req 4.9)."""
    return SystemConfig(
        ollama=OllamaConfig(endpoint_url="http://localhost:11434", model="test-model"),
        workspace_dir=Path("/workspace"),
        max_build_attempts=max_build_attempts,
        concurrency_limit=1,
        context_budget=ContextBudget(max_file_count=10, max_total_bytes=4096),
        allowed_channels=[1],
    )


def _make_target(name: str) -> RegisteredTarget:
    """A target with a build command so the build step runs (Req 4.3)."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("/workspace") / name,
        build_command="make build",
        repo_remote="https://example.invalid/repo.git",
        credentials_ref="GIT_TOKEN_ENV",
    )


def _make_job(target_name: str, idea: str) -> Job:
    return Job(
        id=f"{target_name}-job1",
        target_name=target_name,
        idea=idea,
        status=JobStatus.QUEUED,
        submitted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        channel_id=1,
        user_id=2,
    )


# Feature: discord-ollama-coding-agent, Property 14
# Property 14: Retry is attempt-bounded and feeds prior errors back.
# Validates: Requirements 3.8, 3.9, 4.5, 4.6
@settings(max_examples=20, deadline=None)
@given(
    failure_mode=st.sampled_from(["malformed", "build"]),
    max_build_attempts=st.integers(min_value=1, max_value=6),
    target_name=_SAFE_NAME,
    idea=st.text(min_size=1, max_size=60),
    exit_code=st.integers(min_value=1, max_value=255),
)
@pytest.mark.asyncio
async def test_retry_is_attempt_bounded_and_feeds_prior_errors_back(
    failure_mode: str,
    max_build_attempts: int,
    target_name: str,
    idea: str,
    exit_code: int,
):
    """The loop is attempt-bounded and each retry carries the prior error.

    For a sequence of always-failing attempts the agent must:
    - call generate at most (here, exactly) ``max_build_attempts`` times, never
      more, and complete at most ``max_build_attempts`` attempts (Req 3.8);
    - begin each retry (attempts 2..max) carrying the prior error -- the parse
      error for malformed output, or the previous build's captured output for a
      non-zero build (Req 3.8, 4.5);
    - leave the Job Failed at the bound with the matching reason:
      ``invalid-output`` for malformed output (Req 3.9) or
      ``max-attempts-exhausted`` for a non-zero build (Req 4.6).
    """
    with tempfile.TemporaryDirectory() as working_name:
        working_dir = Path(working_name)

        ollama = _FakeOllamaClient(failure_mode)
        sandbox = _FakeSandbox(working_dir, exit_code)
        target = _make_target(target_name)
        agent = CodingAgent(
            sandbox=sandbox,
            ollama_client=ollama,
            git_manager=_FakeGitManager(),
            config=_make_config(max_build_attempts),
            registry=_FakeRegistry(target),
        )
        job = _make_job(target_name, idea)

        await agent.run(job, CancellationToken())

        # --- Attempt bound: total attempts never exceed the configured max. ---
        assert len(ollama.requests) <= max_build_attempts
        assert job.attempts_completed <= max_build_attempts
        # Every attempt failed, so the loop is driven to its full budget.
        assert len(ollama.requests) == max_build_attempts
        assert job.attempts_completed == max_build_attempts

        # --- Terminal failure with the reason matching the failure domain. ---
        assert job.status is JobStatus.FAILED
        assert job.failure_reason is not None
        if failure_mode == "malformed":
            # Malformed output never parses: exhausted on invalid-output (Req 3.9).
            assert job.failure_reason.startswith(INVALID_OUTPUT_REASON)
            # The build step is never reached when parsing fails.
            assert sandbox.run_calls == 0
        else:
            # A non-zero build through the final attempt (Req 4.6).
            assert job.failure_reason.startswith(MAX_ATTEMPTS_EXHAUSTED_REASON)
            # The build ran once per attempt (each attempt parsed and wrote).
            assert sandbox.run_calls == max_build_attempts

        # --- Feedback chaining: the first attempt carries no prior error... ---
        assert ollama.requests[0].prior_error is None
        # ...and every retry carries the prior attempt's error as feedback.
        for i in range(1, len(ollama.requests)):
            prior_error = ollama.requests[i].prior_error
            assert prior_error is not None and prior_error != ""
            if failure_mode == "malformed":
                # The parse error from the prior attempt is fed back (Req 3.8).
                assert "did not match the expected schema" in prior_error
            else:
                # The prior attempt's exact captured build output (Req 4.5):
                # attempt i+1's request carries build call (i-1)'s output.
                assert prior_error == f"{_BUILD_OUTPUT_PREFIX}::{i - 1}"

        # The idea is preserved unchanged across every retry (only the prior
        # error is refreshed).
        assert all(request.idea == idea for request in ollama.requests)
