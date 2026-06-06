"""Example tests for CodingAgent.run build-branch decisions (task 9.7).

These example/unit tests pin down the two build-branch decisions the agent's
:meth:`CodingAgent.run` loop makes after a successful generation writes files:

- **Build exit 0 proceeds to commit (Req 4.4).** When the Job's
  Registered_Target has a Build_Command and that command exits with status
  zero, the agent proceeds to the commit/push step.
- **No Build_Command skips the build and proceeds to commit (Req 4.7).** When
  the target has no Build_Command, the agent does not run any command and still
  proceeds to the commit/push step.

The agent's collaborators are replaced with small in-memory fakes:

- a fake :class:`OllamaClient` returning a single, valid one-file completion,
- a fake :class:`ExecutionSandbox` whose ``run_command`` returns exit 0 and
  which records whether it was called,
- a fake :class:`GitManager` that records whether ``commit_and_push`` was
  invoked and reports a successful push.

so the decisions are observed through real agent control flow without touching
the network, the filesystem, or Git.

**Validates: Requirements 4.4, 4.7**
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from discord_ollama_agent.coding_agent import CodingAgent
from discord_ollama_agent.execution_sandbox import CancellationToken, CommandResult
from discord_ollama_agent.git_manager import PushResult, PushStatus, SyncResult
from discord_ollama_agent.models.generation import GenerationResult
from discord_ollama_agent.models.job import Job, JobStatus
from discord_ollama_agent.models.target import RegisteredTarget
from discord_ollama_agent.ollama_client import RawCompletion


def _make_job() -> Job:
    """A fresh Queued, non-revision Job to drive through ``run``."""
    return Job(
        id="demo-target-0001",
        target_name="demo-target",
        idea="add a greeting module",
        submitted_at=datetime.now(timezone.utc),
        channel_id=42,
        user_id=7,
    )


def _make_target(build_command: str | None) -> RegisteredTarget:
    """A Registered_Target with the given (optional) Build_Command."""
    return RegisteredTarget(
        name="demo-target",
        directory_path=Path("/workspace/demo-target"),
        build_command=build_command,
        repo_remote="https://example.invalid/demo-target.git",
        credentials_ref="DEMO_TARGET_TOKEN",
    )


class _FakeRegistry:
    """Resolves the single target the Job references."""

    def __init__(self, target: RegisteredTarget) -> None:
        self._target = target

    def get(self, name: str) -> RegisteredTarget | None:
        return self._target if name == self._target.name else None


class _FakeOllamaClient:
    """Returns one valid one-file Generation_Result completion."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request) -> RawCompletion:
        self.calls += 1
        result = GenerationResult.model_validate(
            {"files": [{"path": "greeting.py", "content": "print('hi')\n"}]}
        )
        return RawCompletion(content=result.model_dump_json())


class _FakeSandbox:
    """Records writes and whether/which Build_Command was run (exit 0)."""

    def __init__(self) -> None:
        self.run_command_calls: list[str] = []
        self.written: list[tuple[str, str]] = []

    def init_working_dir(self, job: Job, source: Path) -> Path:
        # The fakes never read the working copy, so any path is fine.
        return Path("/tmp/fake-working-dir")

    def write_file(self, relative_path: str, content: str) -> None:
        self.written.append((relative_path, content))

    async def run_command(
        self, command: str, cancel: CancellationToken | None, timeout_s: int
    ) -> CommandResult:
        self.run_command_calls.append(command)
        return CommandResult(exit_code=0, output="build ok")


class _FakeGitManager:
    """Records sync/commit calls and reports a successful push."""

    def __init__(self) -> None:
        self.commit_and_push_calls = 0
        self.commit_and_push_revision_calls = 0

    async def sync_default_branch(
        self, target: RegisteredTarget, working_copy: Path
    ) -> SyncResult:
        return SyncResult(ok=True)

    async def commit_and_push(
        self, job: Job, target: RegisteredTarget, working_copy: Path
    ) -> PushResult:
        self.commit_and_push_calls += 1
        return PushResult(status=PushStatus.PUSHED, branch_name=job.id)

    async def commit_and_push_revision(
        self, job: Job, target: RegisteredTarget, working_copy: Path
    ) -> PushResult:
        self.commit_and_push_revision_calls += 1
        return PushResult(status=PushStatus.PUSHED, branch_name=job.branch_name)


def _make_agent(
    target: RegisteredTarget,
    sandbox: _FakeSandbox,
    git_manager: _FakeGitManager,
) -> CodingAgent:
    """Wire a CodingAgent over the in-memory fakes (no config => no budget)."""
    return CodingAgent(
        sandbox=sandbox,
        ollama_client=_FakeOllamaClient(),
        git_manager=git_manager,
        config=None,
        registry=_FakeRegistry(target),
    )


# Feature: discord-ollama-coding-agent
# Validates: Requirements 4.4
async def test_build_exit_zero_proceeds_to_commit():
    """A Build_Command exiting 0 leads to commit_and_push (Req 4.4)."""
    target = _make_target(build_command="pytest -q")
    sandbox = _FakeSandbox()
    git_manager = _FakeGitManager()
    agent = _make_agent(target, sandbox, git_manager)
    job = _make_job()

    await agent.run(job, CancellationToken())

    # The build command was run exactly once and exited 0...
    assert sandbox.run_command_calls == ["pytest -q"]
    # ...so the agent proceeded to the commit step.
    assert git_manager.commit_and_push_calls == 1
    assert git_manager.commit_and_push_revision_calls == 0
    # The successful push drives the Job to a Succeeded terminal state.
    assert job.status == JobStatus.SUCCEEDED
    assert job.branch_name == job.id


# Feature: discord-ollama-coding-agent
# Validates: Requirements 4.7
async def test_no_build_command_skips_build_and_proceeds_to_commit():
    """No Build_Command means no command runs, but commit still happens (Req 4.7)."""
    target = _make_target(build_command=None)
    sandbox = _FakeSandbox()
    git_manager = _FakeGitManager()
    agent = _make_agent(target, sandbox, git_manager)
    job = _make_job()

    await agent.run(job, CancellationToken())

    # The build step was skipped entirely: no command was ever run.
    assert sandbox.run_command_calls == []
    # The generated file was still written before skipping the build.
    assert sandbox.written == [("greeting.py", "print('hi')\n")]
    # ...and the agent still proceeded to the commit step.
    assert git_manager.commit_and_push_calls == 1
    assert job.status == JobStatus.SUCCEEDED


def test_fake_completion_is_a_valid_single_file_result():
    """Guard: the fake completion really is a valid one-file Generation_Result.

    Keeps the build-branch tests honest -- if the completion stopped parsing
    into a single FileEntry the agent would fail before reaching the build
    decision, so this anchors the precondition independently.
    """
    payload = {"files": [{"path": "greeting.py", "content": "print('hi')\n"}]}
    parsed = GenerationResult.model_validate_json(json.dumps(payload))
    assert len(parsed.files) == 1
    assert parsed.files[0].path == "greeting.py"
