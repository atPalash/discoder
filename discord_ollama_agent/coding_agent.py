"""Coding_Agent: executes a single Job's lifecycle.

Orchestrates sync -> sandbox init -> context build -> generate -> parse -> write
-> build -> fix loop -> commit/push, emitting lifecycle events on transitions.

This module currently implements the **context-construction** half of the agent
(:meth:`CodingAgent._build_context` and :meth:`CodingAgent._build_revision_context`);
the full :meth:`CodingAgent.run` build loop is added in a later task.

Context construction has two entry points:

- :meth:`CodingAgent._build_context` walks a Job's working copy and returns the
  existing-file context for the *first* generation attempt. An empty working
  copy yields no context (Req 3.2); a non-empty working copy yields a
  budget-bounded subset of its files -- at most ``max_file_count`` files whose
  combined size is at most ``max_total_bytes`` (Req 3.1, 3.4, 3.5, Properties 9
  and 10).
- :meth:`CodingAgent._build_revision_context` loads the contents of a referenced
  Job's existing branch together with the supplied revision feedback into a
  :class:`~discord_ollama_agent.models.generation.GenerationRequest` (Req 11.2).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from pydantic import ValidationError

from .errors import (
    GenerationTimeout,
    OllamaError,
    PathEscapeError,
    SandboxInitError,
    WriteError,
)
from .execution_sandbox import CancellationToken, ExecutionSandbox
from .git_manager import GitManager, PushStatus
from .models.config import ContextBudget, SystemConfig
from .models.generation import FileEntry, GenerationRequest, GenerationResult
from .models.job import Job, JobEvent, JobStatus
from .models.target import RegisteredTarget
from .ollama_client import OllamaClient
from .target_registry import TargetRegistry

__all__ = [
    "CodingAgent",
    "DEFAULT_MAX_BUILD_ATTEMPTS",
    "INVALID_OUTPUT_REASON",
    "EMPTY_GENERATION_REASON",
    "TIMEOUT_REASON",
    "OLLAMA_ERROR_REASON",
    "WRITE_FAILURE_REASON",
    "MAX_ATTEMPTS_EXHAUSTED_REASON",
    "SANDBOX_INIT_FAILURE_REASON",
    "TARGET_UNRESOLVED_REASON",
]

logger = logging.getLogger(__name__)

#: Fallback maximum generate-and-build attempts when no SystemConfig is supplied
#: (matches the SystemConfig default; Req 4.9).
DEFAULT_MAX_BUILD_ATTEMPTS = 3

#: Terminal failure reason: a completion could never be parsed into a valid
#: Generation_Result within the attempt budget (Req 3.9).
INVALID_OUTPUT_REASON = "invalid-output"
#: Terminal failure reason: a parsed Generation_Result carried zero file entries
#: (Req 3.12).
EMPTY_GENERATION_REASON = "empty-generation"
#: Terminal failure reason: generation, build, or test exceeded its timeout
#: (Req 3.10, 4.8).
TIMEOUT_REASON = "timeout"
#: Terminal failure reason prefix: the Ollama endpoint returned an error
#: response; the returned detail is appended (Req 3.11).
OLLAMA_ERROR_REASON = "error"
#: Terminal failure reason: writing a generated file into the sandbox failed or
#: the path was denied for escaping the working directory (Req 4.2, 4.11).
WRITE_FAILURE_REASON = "write-failure"
#: Terminal failure reason prefix: the build command kept failing through the
#: final attempt; the captured build output is appended (Req 4.6).
MAX_ATTEMPTS_EXHAUSTED_REASON = "max-attempts-exhausted"
#: Terminal failure reason: the per-Job sandbox working directory could not be
#: initialized (Req 4.10).
SANDBOX_INIT_FAILURE_REASON = "sandbox-init-failure"
#: Terminal failure reason: the Job's target could not be resolved from the
#: registry (an internal inconsistency, since target existence is validated at
#: Job creation).
TARGET_UNRESOLVED_REASON = "error"

#: Type of the optional asynchronous sink that receives a :class:`JobEvent` on
#: every Job status transition the agent drives (Req 6.1-6.3).
JobEventSink = Callable[[JobEvent], Awaitable[None]]


def _is_timeout(exc: OllamaError) -> bool:
    """Whether an Ollama failure was a generation timeout (Req 3.10 vs 3.11)."""
    return isinstance(exc, GenerationTimeout)

# Directory names skipped when collecting existing-project context. A working
# copy synced from a target's default branch carries VCS metadata (notably
# ``.git``) that is not project source the model should reason about, so it is
# excluded from the context entirely.
_IGNORED_DIR_NAMES: frozenset[str] = frozenset({".git", ".hg", ".svn"})


class CodingAgent:
    """Executes a single Job's lifecycle.

    This class is being built up across tasks. The context-construction methods
    below are pure with respect to the filesystem inputs they are given (they
    take the working copy and budget as arguments rather than reading instance
    state), so they can be exercised in isolation before the orchestrating
    :meth:`run` loop exists.
    """

    def __init__(
        self,
        sandbox: ExecutionSandbox | None = None,
        ollama_client: OllamaClient | None = None,
        git_manager: GitManager | None = None,
        config: SystemConfig | None = None,
        registry: TargetRegistry | None = None,
        event_sink: JobEventSink | None = None,
    ) -> None:
        """Wire the collaborators the :meth:`run` loop orchestrates.

        Every collaborator is optional so the pure context-construction methods
        (:meth:`_build_context`, :meth:`_build_revision_context`) can still be
        exercised against a bare ``CodingAgent()``; :meth:`run` requires the
        execution collaborators and raises if invoked without them.

        Args:
            sandbox: The per-Job :class:`ExecutionSandbox` confining writes and
                running the Build_Command (Req 4.1, 4.2, 4.3).
            ollama_client: The :class:`OllamaClient` used for generation (Req 3.3).
            git_manager: The :class:`GitManager` performing sync and commit/push
                (Req 4.12, 5.x).
            config: The :class:`SystemConfig` supplying the context budget,
                ``max_build_attempts``, and the execution timeout (Req 4.8, 4.9,
                4.14).
            registry: The :class:`TargetRegistry` resolving a Job's target.
            event_sink: Optional async callback invoked with a :class:`JobEvent`
                on each status transition (Req 6.1-6.3).
        """
        self._sandbox = sandbox
        self._ollama_client = ollama_client
        self._git_manager = git_manager
        self._config = config
        self._registry = registry
        self._event_sink = event_sink

    @property
    def _max_build_attempts(self) -> int:
        """The configured maximum generate-and-build attempts per Job (Req 4.9)."""
        if self._config is not None:
            return self._config.max_build_attempts
        return DEFAULT_MAX_BUILD_ATTEMPTS

    @property
    def _context_budget(self) -> ContextBudget | None:
        """The configured context budget bounding first-attempt context, if any."""
        if self._config is not None:
            return self._config.context_budget
        return None

    @property
    def _execution_timeout_s(self) -> int:
        """The Build_Command execution timeout in seconds (Req 4.8)."""
        if self._config is not None:
            return self._config.execution_timeout_s
        return 300

    async def run(self, job: Job, cancel: CancellationToken) -> None:
        """Execute ``job``'s full lifecycle: sync → init → generate/build → push.

        Drives the Job Execution Flow: transitions the Job to Running and emits a
        started event, syncs the working copy to the target's default branch
        (Req 4.12), initializes the per-Job sandbox (Req 4.10), builds the
        generation context, then runs the bounded generate → parse → write →
        build → fix loop (Req 3.7-3.12, 4.1-4.7) before committing and pushing
        via the :class:`GitManager` (Req 5.x). The :class:`CancellationToken` is
        checked between phases so a cancelled Job stops cleanly without
        committing or pushing (Req 8.5). Every terminal outcome records the
        matching ``failure_reason``/``result_note`` on the Job and emits a
        :class:`JobEvent`.

        The method never raises for an expected failure domain; instead it
        records the reason and returns, leaving the Job in a terminal state.
        """
        if (
            self._sandbox is None
            or self._ollama_client is None
            or self._git_manager is None
        ):
            raise RuntimeError(
                "CodingAgent.run requires a sandbox, ollama_client, and git_manager"
            )

        # Transition to Running and announce it (Req 6.1).
        job.status = JobStatus.RUNNING
        await self._emit(job)

        if self._is_cancelled(job, cancel):
            await self._finish_cancelled(job)
            return

        target = self._resolve_target(job)
        if target is None:
            await self._fail(
                job,
                TARGET_UNRESOLVED_REASON,
                detail=f"target {job.target_name!r} is no longer registered",
            )
            return

        # Phase 1: sync the working copy to the target's default branch (Req 4.12).
        sync = await self._git_manager.sync_default_branch(
            target, self._target_source(target)
        )
        if not sync.ok:
            await self._fail(job, sync.failure_reason or "sync-failure", sync.detail)
            return

        if self._is_cancelled(job, cancel):
            await self._finish_cancelled(job)
            return

        # Phase 2: initialize the per-Job sandbox working directory (Req 4.10).
        try:
            working_dir = self._sandbox.init_working_dir(
                job, self._target_source(target)
            )
        except SandboxInitError as exc:
            await self._fail(job, SANDBOX_INIT_FAILURE_REASON, str(exc))
            return

        if self._is_cancelled(job, cancel):
            await self._finish_cancelled(job)
            return

        # Phase 3: build the initial generation request context.
        request = self._initial_request(job, working_dir)

        # Phase 4: the bounded generate → parse → write → build → fix loop.
        loop_outcome = await self._build_loop(job, target, request, working_dir, cancel)
        if loop_outcome is None:
            # A terminal state was already recorded/emitted inside the loop.
            return

        if self._is_cancelled(job, cancel):
            await self._finish_cancelled(job)
            return

        # Phase 5: commit and push the produced changes (Req 5.x, 11.4).
        await self._commit_and_push(job, target, working_dir)

    async def _build_loop(
        self,
        job: Job,
        target: RegisteredTarget,
        request: GenerationRequest,
        working_dir: Path,
        cancel: CancellationToken,
    ) -> bool | None:
        """Run the attempt-bounded generate/parse/write/build/fix loop.

        Returns ``True`` when the loop produced files that built successfully (or
        the build step was skipped) so the caller proceeds to commit/push.
        Returns ``None`` when a terminal state (failure or cancellation) was
        already recorded and emitted, so the caller stops.

        Each iteration counts as one completed attempt. On a recoverable failure
        (malformed completion or non-zero build) the prior error is folded into
        the next request and a new attempt begins while completed attempts remain
        below ``max_build_attempts``; at the final attempt the matching terminal
        reason is recorded (Req 3.8, 3.9, 4.5, 4.6).
        """
        max_attempts = self._max_build_attempts
        for attempt in range(1, max_attempts + 1):
            if self._is_cancelled(job, cancel):
                await self._finish_cancelled(job)
                return None

            is_final = attempt >= max_attempts

            # Generate (Req 3.3, 3.10, 3.11).
            try:
                completion = await self._ollama_client.generate(request)
            except OllamaError as exc:
                reason = TIMEOUT_REASON if _is_timeout(exc) else OLLAMA_ERROR_REASON
                await self._fail(job, reason, str(exc))
                return None

            job.attempts_completed = attempt

            # Parse the completion into a Generation_Result (Req 3.7).
            result, parse_error = self._parse_completion(completion.content)
            if result is None:
                if is_final:
                    # Exhausted attempts on malformed output (Req 3.9).
                    await self._fail(job, INVALID_OUTPUT_REASON, parse_error)
                    return None
                # Feed the parse error back and retry (Req 3.8).
                request = self._retry_request(request, parse_error or "")
                continue

            # Zero file entries is a terminal empty-generation failure (Req 3.12).
            if not result.files:
                await self._fail(job, EMPTY_GENERATION_REASON)
                return None

            if self._is_cancelled(job, cancel):
                await self._finish_cancelled(job)
                return None

            # Write each whole-file entry into the sandbox (Req 4.1, 4.2, 4.11).
            try:
                self._write_files(result.files)
            except PathEscapeError as exc:
                await self._fail(job, WRITE_FAILURE_REASON, str(exc))
                return None
            except WriteError as exc:
                await self._fail(job, WRITE_FAILURE_REASON, str(exc))
                return None

            if self._is_cancelled(job, cancel):
                await self._finish_cancelled(job)
                return None

            # Build/test step (Req 4.3, 4.4, 4.5, 4.6, 4.7).
            if not target.build_command:
                # No Build_Command: skip the build and proceed to commit (Req 4.7).
                return True

            command_result = await self._sandbox.run_command(
                target.build_command, cancel, self._execution_timeout_s
            )

            if command_result.cancelled or self._is_cancelled(job, cancel):
                await self._finish_cancelled(job)
                return None

            if command_result.timed_out:
                # Build/test exceeded the execution timeout (Req 4.8).
                await self._fail(job, TIMEOUT_REASON, command_result.output)
                return None

            if command_result.exit_code == 0:
                # Build succeeded: proceed to commit (Req 4.4).
                return True

            # Non-zero build exit.
            if is_final:
                # Exhausted attempts; record max-attempts-exhausted with the
                # captured build output (Req 4.6).
                await self._fail(
                    job, MAX_ATTEMPTS_EXHAUSTED_REASON, command_result.output
                )
                return None

            # Feed the build output back and retry (Req 4.5).
            request = self._retry_request(request, command_result.output)

        # Defensive: the loop always returns within the body for any attempt
        # count >= 1, but guard against a zero-attempt configuration.
        await self._fail(job, MAX_ATTEMPTS_EXHAUSTED_REASON)
        return None

    async def _commit_and_push(
        self, job: Job, target: RegisteredTarget, working_dir: Path
    ) -> None:
        """Commit and push the produced changes, recording the terminal outcome.

        Delegates to :meth:`GitManager.commit_and_push_revision` for a revision
        Job (continuing its existing branch) and :meth:`GitManager.commit_and_push`
        otherwise (Req 5.1-5.9, 11.4). The :class:`PushResult` maps onto a
        Succeeded (pushed branch or no-changes note) or Failed (missing
        credentials, push error) terminal state.
        """
        if job.is_revision:
            push = await self._git_manager.commit_and_push_revision(
                job, target, working_dir
            )
        else:
            push = await self._git_manager.commit_and_push(job, target, working_dir)

        if push.status == PushStatus.PUSHED:
            job.branch_name = push.branch_name
            await self._succeed(job, branch_name=push.branch_name)
        elif push.status == PushStatus.NO_CHANGES:
            await self._succeed(job, result_note=push.result_note)
        elif push.status == PushStatus.CANCELLED:
            await self._finish_cancelled(job, note=push.result_note)
        else:
            await self._fail(
                job, push.failure_reason or "push-error", push.detail
            )

    def _initial_request(self, job: Job, working_dir: Path) -> GenerationRequest:
        """Assemble the first-attempt generation request for ``job``.

        A revision loads the existing branch contents plus the feedback (carried
        as the Job idea) into the request (Req 11.2); a normal Job builds
        budget-bounded existing-file context from the working copy (empty copy →
        idea only) (Req 3.1, 3.2, 3.4, 3.5).
        """
        if job.is_revision:
            return self._build_revision_context(
                idea=job.idea,
                branch_files=working_dir,
                feedback=job.idea,
                budget=self._context_budget,
            )
        budget = self._context_budget
        context = self._build_context(working_dir, budget) if budget else []
        return GenerationRequest(idea=job.idea, context_files=context)

    @staticmethod
    def _retry_request(request: GenerationRequest, prior_error: str) -> GenerationRequest:
        """Clone ``request`` carrying ``prior_error`` for the next attempt.

        The idea, context, and any revision feedback are preserved; only the
        ``prior_error`` is refreshed so the next generation addresses the most
        recent parse error or build output (Req 3.8, 4.5).
        """
        return GenerationRequest(
            idea=request.idea,
            context_files=request.context_files,
            prior_error=prior_error,
            feedback=request.feedback,
        )

    @staticmethod
    def _parse_completion(content: str) -> tuple[GenerationResult | None, str | None]:
        """Parse a raw completion into a :class:`GenerationResult`.

        Returns ``(result, None)`` on success and ``(None, error_detail)`` when
        the completion is not valid JSON of the expected shape, so the caller can
        feed the error back as a retry (Req 3.7, 3.8). An empty completion is
        treated as malformed rather than as zero files.
        """
        text = content.strip()
        if not text:
            return None, "the model returned an empty completion"
        try:
            return GenerationResult.model_validate_json(text), None
        except ValidationError as exc:
            return None, f"the completion did not match the expected schema: {exc}"

    def _write_files(self, files: Iterable[FileEntry]) -> None:
        """Write each file entry as a whole-file replacement into the sandbox.

        Each path is validated against the sandbox boundary before writing; an
        escaping path raises :class:`PathEscapeError` and a filesystem failure
        raises :class:`WriteError` (Req 4.1, 4.2, 4.11).
        """
        assert self._sandbox is not None
        for entry in files:
            self._sandbox.write_file(entry.path, entry.content)

    def _resolve_target(self, job: Job) -> RegisteredTarget | None:
        """Resolve the Job's :class:`RegisteredTarget` from the registry."""
        if self._registry is None:
            return None
        return self._registry.get(job.target_name)

    @staticmethod
    def _target_source(target: RegisteredTarget) -> Path:
        """The target's on-disk directory copied into the Job working copy."""
        return target.directory_path

    @staticmethod
    def _is_cancelled(job: Job, cancel: CancellationToken) -> bool:
        """Whether the Job has been signalled to stop (Req 8.4, 8.5)."""
        return cancel.cancelled or job.status == JobStatus.CANCELLED

    async def _emit(self, job: Job) -> None:
        """Emit a :class:`JobEvent` snapshot of ``job`` to the sink, if any."""
        if self._event_sink is None:
            return
        await self._event_sink(JobEvent.from_job(job))

    async def _fail(
        self, job: Job, reason: str, detail: str | None = None
    ) -> None:
        """Record a terminal failure on ``job`` and emit a Failed event.

        ``reason`` is the recorded failure-reason code; ``detail`` (build output,
        parse error, exception text) is appended for operator-facing context
        (Req 3.9-3.12, 4.6, 4.10, 4.11, 4.13).
        """
        job.status = JobStatus.FAILED
        if detail:
            job.failure_reason = f"{reason}: {detail}"
        else:
            job.failure_reason = reason
        logger.info("Job %s failed: %s", job.id, job.failure_reason)
        await self._emit(job)

    async def _succeed(
        self,
        job: Job,
        branch_name: str | None = None,
        result_note: str | None = None,
    ) -> None:
        """Record a terminal success on ``job`` and emit a Succeeded event (Req 5.8)."""
        job.status = JobStatus.SUCCEEDED
        if branch_name is not None:
            job.branch_name = branch_name
        if result_note is not None:
            job.result_note = result_note
        await self._emit(job)

    async def _finish_cancelled(self, job: Job, note: str | None = None) -> None:
        """Record the Cancelled terminal state and emit a Cancelled event (Req 8.5).

        The agent never commits or pushes for a cancelled Job; this only records
        the state and notifies the sink.
        """
        job.status = JobStatus.CANCELLED
        if note is not None:
            job.result_note = note
        await self._emit(job)

    def _build_context(
        self, working_copy: Path, budget: ContextBudget
    ) -> list[FileEntry]:
        """Build the existing-file context for a Job's first generation attempt.

        Walks ``working_copy`` for readable text files and returns them as
        :class:`FileEntry` context, bounded by ``budget``:

        - An **empty** working copy (no readable project files) yields ``[]`` so
          the first generation request carries the idea text alone (Req 3.2,
          Property 10).
        - A **non-empty** working copy yields a subset selected so it contains at
          most ``budget.max_file_count`` files and a combined content size of at
          most ``budget.max_total_bytes`` (Req 3.1, 3.4, 3.5, Property 9). When
          the working copy's total content exceeds the budget, only the subset
          that fits is returned and generation proceeds with that subset
          (Req 3.5).

        Files are considered in a stable, path-sorted order so the selected
        subset is deterministic for a given working copy and budget.

        Args:
            working_copy: The Job's working-copy root to collect context from.
            budget: The configured :class:`ContextBudget` bounding file count and
                total bytes.

        Returns:
            The budget-bounded list of context files, or ``[]`` for an empty
            working copy.
        """
        loaded = self._load_files(working_copy)
        if not loaded:
            return []
        return self._select_within_budget(loaded, budget)

    def _build_revision_context(
        self,
        idea: str,
        branch_files: Path | Iterable[FileEntry],
        feedback: str,
        budget: ContextBudget | None = None,
    ) -> GenerationRequest:
        """Assemble the generation request for a revision.

        Loads the contents of the referenced Job's existing branch and pairs them
        with the supplied revision ``feedback`` so the model revises real branch
        code rather than regenerating from scratch (Req 11.2).

        ``branch_files`` may be either the path to a working copy holding the
        checked-out branch (its readable text files are loaded) or an already
        loaded iterable of :class:`FileEntry`. When ``budget`` is supplied, the
        branch contents are bounded by it exactly as in :meth:`_build_context`,
        keeping the request within the model's context window; when ``budget`` is
        ``None`` the full branch contents are included.

        Args:
            idea: The original idea text the revision continues to pursue.
            branch_files: The branch working-copy path, or pre-loaded branch
                file entries.
            feedback: The revision feedback to carry into the request.
            budget: Optional :class:`ContextBudget` bounding the branch context.

        Returns:
            A :class:`GenerationRequest` carrying the idea, the branch context,
            and the feedback.
        """
        if isinstance(branch_files, Path):
            entries = self._load_files(branch_files)
        else:
            entries = list(branch_files)
        if budget is not None:
            entries = self._select_within_budget(entries, budget)
        return GenerationRequest(idea=idea, context_files=entries, feedback=feedback)

    def _load_files(self, root: Path) -> list[FileEntry]:
        """Collect readable UTF-8 text files beneath ``root`` as ``FileEntry``s.

        Recurses through ``root`` skipping VCS metadata directories
        (:data:`_IGNORED_DIR_NAMES`). Each regular file is read as UTF-8; files
        whose bytes are not valid UTF-8 (binary assets) are skipped because they
        are not source text the model can use as context. Paths are recorded
        relative to ``root`` using POSIX separators, and entries are returned
        sorted by path so downstream selection is deterministic.

        A missing ``root`` or one containing no readable text files yields ``[]``.
        """
        root = Path(root)
        if not root.is_dir():
            return []

        entries: list[FileEntry] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in _IGNORED_DIR_NAMES for part in relative.parts):
                continue
            try:
                content = path.read_bytes().decode("utf-8")
            except (OSError, UnicodeDecodeError):
                # Unreadable or non-text file: not usable as model context.
                continue
            entries.append(FileEntry(path=relative.as_posix(), content=content))

        entries.sort(key=lambda entry: entry.path)
        return entries

    @staticmethod
    def _select_within_budget(
        entries: list[FileEntry], budget: ContextBudget
    ) -> list[FileEntry]:
        """Greedily select a subset of ``entries`` that fits within ``budget``.

        Walks ``entries`` in order, adding each file whose inclusion keeps the
        running selection within both bounds: at most ``budget.max_file_count``
        files and a combined UTF-8 content size of at most
        ``budget.max_total_bytes``. A file larger on its own than the byte budget
        is skipped (it can never fit); selection stops once the file-count bound
        is reached. The result therefore always satisfies both budget limits
        (Req 3.4, 3.5, Property 9).
        """
        selected: list[FileEntry] = []
        total_bytes = 0
        for entry in entries:
            if len(selected) >= budget.max_file_count:
                break
            entry_bytes = len(entry.content.encode("utf-8"))
            if total_bytes + entry_bytes > budget.max_total_bytes:
                continue
            selected.append(entry)
            total_bytes += entry_bytes
        return selected
