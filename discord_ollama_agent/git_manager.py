"""Git_Manager: per-Job Git operations against a target's repository.

Performs the Git side of a Job's lifecycle for the Job's Registered_Target,
using that target's own remote, branch-naming scheme, and credentials:

- :meth:`GitManager.sync_default_branch` -- fetches the target's default branch
  from its remote and resets the Job's working copy to it, so generation is
  based on current code; a fetch/reset failure maps to a ``sync-failure`` reason
  (Req 4.12, 4.13).
- :meth:`GitManager.branch_name_for` -- derives the branch name from the Job id
  via the target's ``branch_scheme``; because the Job id contains the target
  name, the branch name does too (Req 5.1).
- :meth:`GitManager.commit_and_push` -- creates a branch from the synced state,
  stages every add/modify/delete, commits referencing the Job id, and pushes
  with the target's credentials within the push timeout. A working copy with no
  changes yields a Succeeded no-changes note with no commit/push (Req 5.6);
  unresolved credentials yield ``missing-credentials`` with no push (Req 5.7); a
  Cancelled Job is refused entirely (Req 5.9); a failed/timed-out push maps to a
  ``push-error`` reason (Req 5.5). On success the pushed branch name is returned
  (Req 5.1-5.4, 5.8).
- :meth:`GitManager.commit_and_push_revision` -- adds new commits to the
  referenced Job's existing branch and pushes them (Req 11.4).

Credentials are resolved at runtime from the target's ``credentials_ref`` -- an
environment-variable name or a secret-file path -- never from a stored secret
value. The resolved secret is injected per invocation via a ``GIT_ASKPASS``
helper passed through the subprocess environment; it is never embedded in a URL,
written to the registry, or logged (Req 5.4, 5.7).

Blocking GitPython calls run via :func:`asyncio.to_thread` so the event loop
stays responsive, and network ``fetch``/``push`` operations carry a timeout that
terminates the underlying ``git`` subprocess (the Concurrency Model in the
design).
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, PushInfo, Repo

from .models.job import Job, JobStatus
from .models.target import RegisteredTarget

__all__ = [
    "SYNC_FAILURE_REASON",
    "MISSING_CREDENTIALS_REASON",
    "PUSH_ERROR_REASON",
    "SyncResult",
    "PushStatus",
    "PushResult",
    "resolve_credentials",
    "GitManager",
]

logger = logging.getLogger(__name__)

#: Failure reason recorded when fetching/resetting the default branch fails
#: (Req 4.13).
SYNC_FAILURE_REASON = "sync-failure"
#: Failure reason recorded when the target's credentials reference resolves to
#: nothing; no push is attempted (Req 5.7).
MISSING_CREDENTIALS_REASON = "missing-credentials"
#: Failure reason recorded when the push fails or exceeds the push timeout
#: (Req 5.5).
PUSH_ERROR_REASON = "push-error"

#: Default username presented to the remote when the resolved credential is a
#: bare token (works for token-as-password HTTPS auth, e.g. GitHub/GitLab).
_DEFAULT_CREDENTIAL_USERNAME = "x-access-token"

#: Small buffer added to the configured timeout for the ``asyncio.wait_for``
#: backstop, so the ``git`` subprocess's own ``kill_after_timeout`` is the
#: primary enforcement mechanism.
_TIMEOUT_BACKSTOP_BUFFER_S = 5.0


@dataclass(frozen=True)
class SyncResult:
    """Outcome of syncing a working copy to its target's default branch.

    Attributes:
        ok: Whether the working copy was successfully reset to the latest
            default-branch state from the remote.
        failure_reason: The recorded reason code when ``ok`` is ``False`` (always
            :data:`SYNC_FAILURE_REASON`); ``None`` on success (Req 4.13).
        detail: A short, secret-free description of the failure for the operator
            log; ``None`` on success.
    """

    ok: bool
    failure_reason: str | None = None
    detail: str | None = None


class PushStatus(str, Enum):
    """The terminal outcome of a commit/push attempt.

    ``PUSHED`` and ``NO_CHANGES`` both correspond to a Succeeded Job (a pushed
    branch or a no-changes note); ``FAILED`` corresponds to a Failed Job;
    ``CANCELLED`` means the Git_Manager refused to act because the Job was
    Cancelled and performed no staging/commit/push (Req 5.6, 5.7, 5.8, 5.9).
    """

    PUSHED = "pushed"
    NO_CHANGES = "no-changes"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PushResult:
    """Outcome of :meth:`GitManager.commit_and_push` / ``commit_and_push_revision``.

    Attributes:
        status: The :class:`PushStatus` describing what happened.
        branch_name: The pushed branch name when ``status`` is ``PUSHED``;
            ``None`` otherwise (Req 5.1, 5.8).
        result_note: A human-readable note for a Succeeded Job -- the no-changes
            note when ``status`` is ``NO_CHANGES`` (Req 5.6).
        failure_reason: The recorded reason code when ``status`` is ``FAILED``
            (:data:`MISSING_CREDENTIALS_REASON` or :data:`PUSH_ERROR_REASON`).
        detail: A short, secret-free description of the failure for the operator
            log; ``None`` unless ``status`` is ``FAILED``.
    """

    status: PushStatus
    branch_name: str | None = None
    result_note: str | None = None
    failure_reason: str | None = None
    detail: str | None = None


def resolve_credentials(credentials_ref: str | None) -> str | None:
    """Resolve a target's credentials reference to a secret value at runtime.

    The reference is interpreted as either the name of an environment variable
    holding the secret or the path to a secret file. The first that yields a
    non-empty value wins; otherwise ``None`` is returned to signal that no
    credentials are present (Req 5.4, 5.7).

    The resolved secret value is never logged or persisted; only its presence or
    absence is observable to callers.

    Args:
        credentials_ref: The environment-variable name or secret-file path stored
            on the Registered_Target, or ``None``.

    Returns:
        The resolved secret string with surrounding whitespace stripped, or
        ``None`` when the reference is empty or resolves to nothing.
    """
    if not credentials_ref or not credentials_ref.strip():
        return None

    env_value = os.environ.get(credentials_ref)
    if env_value and env_value.strip():
        return env_value.strip()

    try:
        candidate = Path(credentials_ref)
        if candidate.is_file():
            file_value = candidate.read_text(encoding="utf-8").strip()
            if file_value:
                return file_value
    except OSError:
        # An unreadable secret path is treated as "no credentials present";
        # the value (and any path error detail) is never logged.
        return None

    return None


@contextmanager
def _credential_environment(repo: Repo, secret: str | None) -> Iterator[None]:
    """Inject ``secret`` into ``repo``'s git subprocess environment for one block.

    When a secret is present, a short-lived ``GIT_ASKPASS`` helper script is
    written that echoes the credential from the environment (never from the
    command line or a URL), so the secret is supplied to the remote without being
    logged or persisted. ``GIT_TERMINAL_PROMPT=0`` ensures git never blocks on an
    interactive prompt when no/invalid credentials are supplied. When no secret
    is present the environment is left unchanged (public remotes still work; a
    private remote will simply fail the push, which maps to ``push-error``).

    The helper script is removed when the block exits.
    """
    if secret is None:
        with repo.git.custom_environment(GIT_TERMINAL_PROMPT="0"):
            yield
        return

    fd, script_name = tempfile.mkstemp(prefix=".git-askpass-", suffix=".sh")
    script_path = Path(script_name)
    try:
        # The helper reads the credential from the environment so the secret
        # never appears on a command line or in the script body on disk.
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '  *[Uu]sername*) printf %s \"$GIT_AGENT_USERNAME\" ;;\n'
                '  *) printf %s \"$GIT_AGENT_PASSWORD\" ;;\n'
                "esac\n"
            )
        script_path.chmod(stat.S_IRWXU)

        with repo.git.custom_environment(
            GIT_ASKPASS=str(script_path),
            GIT_TERMINAL_PROMPT="0",
            GIT_AGENT_USERNAME=_DEFAULT_CREDENTIAL_USERNAME,
            GIT_AGENT_PASSWORD=secret,
        ):
            yield
    finally:
        script_path.unlink(missing_ok=True)


class GitManager:
    """Per-Job Git operations using a target's remote, branch scheme, and creds.

    The manager is stateless across Jobs; the working copy, target, and Job are
    supplied to each call. A single configured push timeout bounds the network
    ``fetch``/``push`` operations.
    """

    def __init__(self, push_timeout_s: int = 120) -> None:
        """Create a Git_Manager.

        Args:
            push_timeout_s: Maximum time, in seconds, allowed for a network
                ``fetch``/``push`` before the underlying ``git`` subprocess is
                terminated and the operation is treated as failed (Req 5.5).
        """
        self._push_timeout_s = push_timeout_s

    def branch_name_for(self, job: Job, target: RegisteredTarget) -> str:
        """Derive the branch name for ``job`` from the target's branch scheme.

        The branch name is produced by formatting ``target.branch_scheme`` with
        the Job id (and the target name, for schemes that reference it). Because
        the Job id contains the target name, the resulting branch name includes
        the target name too (Req 5.1). A scheme that references an unknown
        placeholder falls back to the Job id so branch derivation never fails.

        Args:
            job: The Job whose id seeds the branch name.
            target: The Registered_Target providing the branch-naming scheme.

        Returns:
            The derived branch name.
        """
        try:
            return target.branch_scheme.format(job_id=job.id, target_name=target.name)
        except (KeyError, IndexError, ValueError):
            logger.warning(
                "Branch scheme %r for target %r is invalid; falling back to the Job id",
                target.branch_scheme,
                target.name,
            )
            return job.id

    async def sync_default_branch(
        self, target: RegisteredTarget, working_copy: Path
    ) -> SyncResult:
        """Reset ``working_copy`` to the target's default branch from its remote.

        Fetches the target's ``default_branch`` from ``repo_remote`` and hard-resets
        the working copy to it, so generation starts from current code (Req 4.12).
        If the working copy is not yet a Git repository it is cloned from the
        remote. Any failure -- unreachable remote, missing branch, fetch timeout,
        or reset error -- is mapped to a :data:`SYNC_FAILURE_REASON` result
        (Req 4.13). Credentials, when configured, are injected for the fetch so a
        private remote can be reached.

        Args:
            target: The Job's Registered_Target.
            working_copy: The Job's isolated working-copy root.

        Returns:
            A :class:`SyncResult` indicating success or a sync failure.
        """
        secret = resolve_credentials(target.credentials_ref)
        try:
            return await self._run_with_timeout(
                self._sync_default_branch_blocking, target, working_copy, secret
            )
        except asyncio.TimeoutError:
            return SyncResult(
                ok=False,
                failure_reason=SYNC_FAILURE_REASON,
                detail=f"fetch exceeded the {self._push_timeout_s}s timeout",
            )
        except (GitCommandError, OSError, ValueError) as exc:
            return SyncResult(
                ok=False,
                failure_reason=SYNC_FAILURE_REASON,
                detail=self._safe_git_detail(exc),
            )

    def _sync_default_branch_blocking(
        self, target: RegisteredTarget, working_copy: Path, secret: str | None
    ) -> SyncResult:
        """Blocking implementation of :meth:`sync_default_branch` (runs in a thread)."""
        default_branch = target.default_branch
        try:
            repo = Repo(working_copy)
        except (InvalidGitRepositoryError, NoSuchPathError):
            working_copy.mkdir(parents=True, exist_ok=True)
            repo = self._clone(target, working_copy, secret)
            return SyncResult(ok=True)

        origin = self._ensure_origin(repo, target.repo_remote)
        with _credential_environment(repo, secret):
            origin.fetch(default_branch, kill_after_timeout=self._push_timeout_s)
        # Reset the working tree to exactly the fetched default-branch tip.
        repo.git.reset("--hard", "FETCH_HEAD")
        return SyncResult(ok=True)

    def _clone(
        self, target: RegisteredTarget, working_copy: Path, secret: str | None
    ) -> Repo:
        """Clone the target's remote into ``working_copy`` (used for a fresh copy)."""
        env: dict[str, str] = {"GIT_TERMINAL_PROMPT": "0"}
        cleanup: Path | None = None
        if secret is not None:
            cleanup = self._write_askpass()
            env.update(
                GIT_ASKPASS=str(cleanup),
                GIT_AGENT_USERNAME=_DEFAULT_CREDENTIAL_USERNAME,
                GIT_AGENT_PASSWORD=secret,
            )
        try:
            return Repo.clone_from(
                target.repo_remote,
                str(working_copy),
                branch=target.default_branch,
                env=env,
                kill_after_timeout=self._push_timeout_s,
            )
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)

    async def commit_and_push(
        self, job: Job, target: RegisteredTarget, working_copy: Path
    ) -> PushResult:
        """Create a branch, commit all changes, and push for ``job`` (Req 5.1-5.9).

        Refuses to act on a Cancelled Job (no staging/commit/push) (Req 5.9). When
        the working copy has no differences from its synced state, returns a
        no-changes Succeeded result without committing or pushing (Req 5.6). When
        the target's credentials cannot be resolved, returns
        :data:`MISSING_CREDENTIALS_REASON` without pushing (Req 5.7). Otherwise it
        creates a branch from the synced state whose name derives from the Job id
        (Req 5.1), stages all added/modified/deleted files, commits with a message
        referencing the Job id (Req 5.2), and pushes the branch to the target's
        remote using the resolved credentials within the push timeout (Req 5.3,
        5.4, 5.5). A successful push yields a Succeeded result carrying the pushed
        branch name (Req 5.8).

        Args:
            job: The Job whose changes are being committed.
            target: The Job's Registered_Target.
            working_copy: The Job's isolated working-copy root.

        Returns:
            A :class:`PushResult` describing the outcome.
        """
        branch_name = self.branch_name_for(job, target)
        return await self._commit_and_push(
            job, target, working_copy, branch_name, new_branch=True
        )

    async def commit_and_push_revision(
        self, job: Job, target: RegisteredTarget, working_copy: Path
    ) -> PushResult:
        """Add commits to the referenced Job's existing branch and push (Req 11.4).

        Behaves like :meth:`commit_and_push` but continues the Job's existing
        branch (``job.branch_name``) rather than creating a new one: it checks out
        that branch, stages all changes, commits referencing the Job id, and
        pushes the new commits to the target's remote. The same Cancelled,
        no-changes, and missing-credentials guards apply.

        Args:
            job: The revision Job, whose ``branch_name`` names the branch to
                continue.
            target: The Job's Registered_Target.
            working_copy: The Job's isolated working-copy root.

        Returns:
            A :class:`PushResult` describing the outcome.

        Raises:
            ValueError: If ``job.branch_name`` is not set (a revision must
                reference an existing branch).
        """
        if not job.branch_name:
            raise ValueError(
                "commit_and_push_revision requires the Job to carry a branch_name"
            )
        return await self._commit_and_push(
            job, target, working_copy, job.branch_name, new_branch=False
        )

    async def _commit_and_push(
        self,
        job: Job,
        target: RegisteredTarget,
        working_copy: Path,
        branch_name: str,
        *,
        new_branch: bool,
    ) -> PushResult:
        """Shared commit/push flow for both new branches and revisions."""
        # Req 5.9 / Property 16: a Cancelled Job never stages, commits, or pushes.
        if job.status == JobStatus.CANCELLED:
            return PushResult(
                status=PushStatus.CANCELLED,
                result_note="job cancelled; no changes were committed or pushed",
            )

        secret = resolve_credentials(target.credentials_ref)
        try:
            return await self._run_with_timeout(
                self._commit_and_push_blocking,
                job,
                target,
                working_copy,
                branch_name,
                new_branch,
                secret,
            )
        except asyncio.TimeoutError:
            return PushResult(
                status=PushStatus.FAILED,
                failure_reason=PUSH_ERROR_REASON,
                detail=f"push exceeded the {self._push_timeout_s}s timeout",
            )
        except (GitCommandError, OSError, ValueError) as exc:
            return PushResult(
                status=PushStatus.FAILED,
                failure_reason=PUSH_ERROR_REASON,
                detail=self._safe_git_detail(exc),
            )

    def _commit_and_push_blocking(
        self,
        job: Job,
        target: RegisteredTarget,
        working_copy: Path,
        branch_name: str,
        new_branch: bool,
        secret: str | None,
    ) -> PushResult:
        """Blocking implementation of the commit/push flow (runs in a thread)."""
        repo = Repo(working_copy)

        # Req 5.6 / Property 18: no differences from the synced state -> no-changes.
        if not repo.is_dirty(index=True, working_tree=True, untracked_files=True):
            return PushResult(
                status=PushStatus.NO_CHANGES,
                result_note="no file changes were produced",
            )

        # Req 5.7 / Property 17: unresolved credentials -> no commit, no push.
        if secret is None:
            return PushResult(
                status=PushStatus.FAILED,
                failure_reason=MISSING_CREDENTIALS_REASON,
                detail=(
                    f"credentials reference {target.credentials_ref!r} for target "
                    f"{target.name!r} resolved to no value"
                ),
            )

        self._ensure_origin(repo, target.repo_remote)

        if new_branch:
            # Branch from the current (synced) state (Req 5.1).
            repo.git.checkout("-B", branch_name)
        else:
            # Continue the referenced Job's existing branch (Req 11.4).
            repo.git.checkout(branch_name)

        # Stage every added, modified, and deleted file (Req 5.2).
        repo.git.add("--all")
        # Commit referencing the Job id (Req 5.2 / Property 19).
        repo.index.commit(f"Apply changes for Job {job.id}")

        with _credential_environment(repo, secret):
            push_infos = repo.remote("origin").push(
                refspec=f"{branch_name}:{branch_name}",
                kill_after_timeout=self._push_timeout_s,
            )

        self._raise_on_push_error(push_infos)
        return PushResult(status=PushStatus.PUSHED, branch_name=branch_name)

    @staticmethod
    def _ensure_origin(repo: Repo, remote_url: str):
        """Return the ``origin`` remote, creating/retargeting it to ``remote_url``."""
        if "origin" in {remote.name for remote in repo.remotes}:
            origin = repo.remote("origin")
            if remote_url and remote_url not in list(origin.urls):
                origin.set_url(remote_url)
            return origin
        return repo.create_remote("origin", remote_url)

    @staticmethod
    def _raise_on_push_error(push_infos) -> None:
        """Raise :class:`GitCommandError` if any push ref reported an error flag.

        GitPython does not always raise on a rejected/failed push; the per-ref
        flags must be inspected so a rejection still maps to ``push-error``.
        """
        error_mask = PushInfo.ERROR | PushInfo.REJECTED | PushInfo.REMOTE_REJECTED
        for info in push_infos:
            if info.flags & error_mask:
                raise GitCommandError(
                    "git push", 1, stderr=(info.summary or "push rejected").strip()
                )

    async def _run_with_timeout(self, func, *args):
        """Run a blocking git ``func`` in a thread, bounded by the push timeout.

        ``git``'s own ``kill_after_timeout`` is the primary enforcement (it
        terminates the subprocess); :func:`asyncio.wait_for` is a slightly longer
        backstop so a hung thread cannot block indefinitely.
        """
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args),
            timeout=self._push_timeout_s + _TIMEOUT_BACKSTOP_BUFFER_S,
        )

    @staticmethod
    def _write_askpass() -> Path:
        """Write a standalone ``GIT_ASKPASS`` helper script and return its path."""
        fd, script_name = tempfile.mkstemp(prefix=".git-askpass-", suffix=".sh")
        script_path = Path(script_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '  *[Uu]sername*) printf %s \"$GIT_AGENT_USERNAME\" ;;\n'
                '  *) printf %s \"$GIT_AGENT_PASSWORD\" ;;\n'
                "esac\n"
            )
        script_path.chmod(stat.S_IRWXU)
        return script_path

    @staticmethod
    def _safe_git_detail(exc: Exception) -> str:
        """Render a short, secret-free description of a Git/OS failure.

        Only the exception type and its message are surfaced. Credential values
        are never placed in command lines or URLs by this module, so they cannot
        appear here.
        """
        message = str(exc).strip() or exc.__class__.__name__
        # Collapse to a single line to keep operator-log entries tidy.
        return message.splitlines()[0]
