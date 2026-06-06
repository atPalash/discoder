"""Execution_Sandbox: per-Job working-directory boundary inside the container.

Confines all file writes to the Job's working directory (path traversal denied)
and runs build/test commands under timeouts with terminable process groups. The
entire System runs inside a single container; the sandbox does **not** spin up a
container per Job. Instead each Job gets its own isolated working-copy root
inside the shared container (Req 7.2, 7.7), writes are path-confined to that
root (Req 4.1, 4.2, 4.11), and build/test commands run under an execution
timeout with output capture capped at the configured maximum (Req 4.3, 4.6,
4.8).

The component exposes four operations matching the design contract:

- :meth:`ExecutionSandbox.init_working_dir` creates the distinct per-Job
  working-copy root from a source directory, raising
  :class:`~discord_ollama_agent.errors.SandboxInitError` on failure (Req 4.10,
  7.2).
- :meth:`ExecutionSandbox.resolve_within` joins a relative path to the working
  dir, fully resolves it (normalizing ``..`` and following symlinks), and
  verifies it stays inside the working dir, raising
  :class:`~discord_ollama_agent.errors.PathEscapeError` on escape (Req 4.2).
- :meth:`ExecutionSandbox.write_file` performs a path-validated whole-file
  write, raising :class:`~discord_ollama_agent.errors.WriteError` on failure
  (Req 4.1, 4.11).
- :meth:`ExecutionSandbox.run_command` runs a build/test command in the working
  dir under a new process group, captures combined output capped at
  ``build_output_cap_bytes``, and terminates the whole process tree on timeout
  or cancellation (Req 4.3, 4.6, 4.8).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import PathEscapeError, SandboxInitError, WriteError
from .models.config import DEFAULT_BUILD_OUTPUT_CAP_BYTES
from .models.job import Job

__all__ = [
    "CancellationToken",
    "CommandResult",
    "ExecutionSandbox",
]

# How long, in seconds, to wait after a SIGTERM before escalating to SIGKILL
# when terminating a build/test process group (timeout or cancellation).
_TERMINATION_GRACE_S = 5.0

# Size of each read from the subprocess output stream.
_READ_CHUNK_SIZE = 65_536

# Characters allowed verbatim in a working-directory name prefix; everything
# else (including path separators that could appear in a Job id's target name)
# is replaced so the prefix can never alter the directory layout.
_SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._-]")


class CancellationToken:
    """Cooperative stop signal shared between a Job worker and the sandbox.

    Wraps an :class:`asyncio.Event` so the orchestration layer can signal a Job
    to stop (``/cancel`` against a Running Job) and :meth:`ExecutionSandbox.run_command`
    can react by terminating the active build/test process tree (Req 8.4). The
    token is one-shot: once cancelled it stays cancelled.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Signal cancellation. Idempotent."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Block until cancellation is requested (returns immediately if already)."""
        await self._event.wait()


@dataclass(frozen=True)
class CommandResult:
    """The outcome of running a build/test command in the sandbox.

    Attributes:
        exit_code: The process exit status, or ``None`` if the process was
            terminated before reporting one.
        output: Combined stdout/stderr captured from the command, decoded as
            UTF-8 (undecodable bytes replaced) and truncated to the configured
            output cap (Req 4.6).
        timed_out: ``True`` if the command was terminated for exceeding the
            execution timeout (Req 4.8).
        cancelled: ``True`` if the command was terminated because the Job was
            cancelled (Req 8.4).
        output_truncated: ``True`` if the command produced more output than the
            configured cap, so ``output`` is a prefix of the full output.
    """

    exit_code: int | None
    output: str
    timed_out: bool = False
    cancelled: bool = False
    output_truncated: bool = False


class ExecutionSandbox:
    """Per-Job working-directory boundary for writes and command execution.

    A sandbox instance is created per Job and bound to a single working-copy
    root by :meth:`init_working_dir`. All subsequent operations are confined to
    that root: :meth:`resolve_within`/:meth:`write_file` deny any path that
    escapes it (Req 4.2), and :meth:`run_command` runs with the working dir as
    its current directory (Req 4.3).

    Args:
        root_dir: Parent directory under which per-Job working-copy roots are
            created. Distinct Jobs always get distinct roots beneath it
            (Req 7.2, 7.7).
        build_output_cap_bytes: Maximum number of bytes of command output
            retained by :meth:`run_command`; defaults to 1 MiB (Req 4.6).
    """

    def __init__(
        self,
        root_dir: Path,
        build_output_cap_bytes: int = DEFAULT_BUILD_OUTPUT_CAP_BYTES,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._output_cap_bytes = int(build_output_cap_bytes)
        self._working_dir: Path | None = None
        self._working_dir_resolved: Path | None = None

    @property
    def working_dir(self) -> Path:
        """The Job's working-copy root.

        Raises:
            RuntimeError: If accessed before :meth:`init_working_dir`.
        """
        if self._working_dir is None:
            raise RuntimeError("ExecutionSandbox.init_working_dir has not been called")
        return self._working_dir

    def init_working_dir(self, job: Job, source: Path) -> Path:
        """Create this Job's isolated working-copy root from ``source``.

        Creates a fresh directory beneath ``root_dir`` whose name is derived from
        the (unique, never-reused) Job id, guaranteeing it is distinct from every
        other Job's root including other Jobs against the same target (Req 7.2,
        7.7). The contents of ``source`` (the target's directory) are copied into
        the new root so the Job operates on its own working copy (Req 1.9, 7.2);
        a missing or empty ``source`` yields an empty working copy, which is the
        valid "new code" case.

        Args:
            job: The Job this sandbox is for; its id seeds the root's name.
            source: The target directory whose contents are copied in.

        Returns:
            The path to the created working-copy root.

        Raises:
            SandboxInitError: If the working directory cannot be created or the
                source contents cannot be copied (Req 4.10).
        """
        try:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            prefix = _SAFE_PREFIX_RE.sub("_", job.id)
            # mkdtemp creates a fresh, uniquely named directory atomically, so
            # distinctness holds even if two Jobs share a target/id-prefix.
            working = Path(
                tempfile.mkdtemp(prefix=f"{prefix}-", dir=str(self._root_dir))
            )
            source_path = Path(source)
            if source_path.is_dir():
                shutil.copytree(source_path, working, dirs_exist_ok=True)
        except (OSError, shutil.Error) as exc:
            raise SandboxInitError(
                f"could not initialize sandbox working directory for Job {job.id!r}: "
                f"{exc}"
            ) from exc

        self._working_dir = working
        # Resolve once so containment checks compare against a canonical root
        # (the root itself may live under a symlinked path such as /tmp).
        self._working_dir_resolved = working.resolve()
        return working

    def resolve_within(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` against the working dir, denying escapes.

        Joins ``relative_path`` to the working-dir root, fully resolves the
        result (normalizing ``..`` and following symlinks), and verifies it is
        relative to the resolved working-dir root. Absolute paths, ``..``
        traversal, and symlink escapes are all rejected because the check runs on
        the canonical resolved path (Req 4.2).

        Args:
            relative_path: A path interpreted relative to the working dir.

        Returns:
            The resolved, in-bounds absolute path.

        Raises:
            RuntimeError: If called before :meth:`init_working_dir`.
            PathEscapeError: If the path resolves outside the working dir.
        """
        if self._working_dir is None or self._working_dir_resolved is None:
            raise RuntimeError("ExecutionSandbox.init_working_dir has not been called")

        # An absolute ``relative_path`` resets the join to itself; resolve() then
        # exposes it as outside the working dir, so the containment check below
        # still rejects it.
        candidate = self._working_dir / relative_path
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._working_dir_resolved):
            raise PathEscapeError(
                f"path {relative_path!r} resolves to {str(resolved)!r}, which is "
                f"outside the Job working directory {str(self._working_dir)!r}"
            )
        return resolved

    def write_file(self, relative_path: str, content: str) -> None:
        """Write ``content`` to ``relative_path`` as a whole-file replacement.

        The path is validated against the sandbox boundary first; an escaping
        path raises :class:`PathEscapeError` and no file outside the working dir
        is touched (Req 4.2). Otherwise the file is created (with any missing
        parent directories) or overwritten with ``content`` (Req 4.1). Any
        filesystem failure is surfaced as :class:`WriteError` (Req 4.11).

        Args:
            relative_path: Destination path relative to the working dir.
            content: Full intended file contents (whole-file write).

        Raises:
            RuntimeError: If called before :meth:`init_working_dir`.
            PathEscapeError: If the path resolves outside the working dir.
            WriteError: If creating directories or writing the file fails.
        """
        # PathEscapeError from resolve_within propagates unwrapped: an escape is
        # a denied write (Req 4.2), distinct from a write-failure (Req 4.11).
        resolved = self.resolve_within(relative_path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise WriteError(
                f"failed to write file {relative_path!r} into the Job working "
                f"directory: {exc}"
            ) from exc

    async def run_command(
        self,
        command: str,
        cancel: CancellationToken | None,
        timeout_s: int,
    ) -> CommandResult:
        """Run ``command`` in the working dir under a terminable process group.

        Launches ``command`` via a shell with the working dir as its current
        directory (Req 4.3) and in a new session, so the whole process tree can
        be terminated as a group. Combined stdout/stderr is captured but never
        retained beyond ``build_output_cap_bytes`` (Req 4.6); excess output is
        still drained so the process cannot block on a full pipe. If the command
        exceeds ``timeout_s`` it is terminated and the result is flagged as
        timed out (Req 4.8); if ``cancel`` is signalled the process tree is
        likewise terminated (Req 8.4).

        Args:
            command: The shell command line to run (the target's Build_Command).
            cancel: Cancellation token to abort the command, or ``None``.
            timeout_s: Maximum runtime in seconds before termination (Req 4.8).

        Returns:
            A :class:`CommandResult` with the exit code, captured output, and
            timeout/cancellation flags.

        Raises:
            RuntimeError: If called before :meth:`init_working_dir`.
        """
        if self._working_dir is None:
            raise RuntimeError("ExecutionSandbox.init_working_dir has not been called")

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self._working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            # New session => new process group, so os.killpg terminates the whole
            # tree the command spawns (Req 4.8, 8.4).
            start_new_session=True,
        )

        captured = bytearray()
        truncated = False

        async def _drain_output() -> None:
            nonlocal truncated
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(_READ_CHUNK_SIZE)
                if not chunk:
                    break
                remaining = self._output_cap_bytes - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True
                else:
                    # Cap reached; keep draining to avoid a full-pipe deadlock
                    # but discard the overflow (Req 4.6).
                    truncated = True

        reader_task = asyncio.ensure_future(_drain_output())
        wait_task = asyncio.ensure_future(proc.wait())
        waiters: set[asyncio.Future] = {wait_task}
        cancel_task: asyncio.Future | None = None
        if cancel is not None:
            cancel_task = asyncio.ensure_future(cancel.wait())
            waiters.add(cancel_task)

        timed_out = False
        cancelled = False
        try:
            done, _pending = await asyncio.wait(
                waiters,
                timeout=timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task not in done:
                # Either the timeout elapsed (nothing completed) or cancellation
                # fired first; in both cases terminate the process tree.
                if cancel_task is not None and cancel_task in done:
                    cancelled = True
                else:
                    timed_out = True
                await self._terminate(proc)
        finally:
            if cancel_task is not None and not cancel_task.done():
                cancel_task.cancel()
                try:
                    await cancel_task
                except asyncio.CancelledError:
                    pass

        exit_code = await wait_task
        await reader_task

        output = bytes(captured).decode("utf-8", errors="replace")
        return CommandResult(
            exit_code=exit_code,
            output=output,
            timed_out=timed_out,
            cancelled=cancelled,
            output_truncated=truncated,
        )

    @staticmethod
    async def _terminate(proc: asyncio.subprocess.Process) -> None:
        """Terminate a process group: SIGTERM, then SIGKILL after a grace period.

        Sends the signal to the process group (created via ``start_new_session``)
        so any children the command spawned are also killed. Already-exited
        processes and missing groups are ignored.
        """
        if proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=_TERMINATION_GRACE_S)
        except asyncio.TimeoutError:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
