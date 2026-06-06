"""Target_Registry: the persisted store of Registered_Targets.

Owns the JSON-persisted set of :class:`~discord_ollama_agent.models.target.RegisteredTarget`.
Its three responsibilities map onto the design's contract:

- **Load + partition (Req 12.1-12.3).** :meth:`TargetRegistry.load` reads the
  persisted registry and splits it into a *usable* set and an *excluded* set.
  A target is usable iff it has the required per-target configuration -- a name
  (1..64 non-whitespace Unicode characters, enforced by the model), a
  ``directory_path`` that resolves inside the Workspace_Directory, and a
  non-empty Target_Repository ``repo_remote``. Every excluded target is recorded
  in the operator log with its name and the missing/offending value, so Jobs
  only ever run against usable targets (Req 1.6, 12.4).
- **Resolve (Req 1.1, 9.1, 9.2, 12.4).** :meth:`get`, :meth:`names`, and
  :meth:`exists` answer questions about the registry. ``get``/``names`` consult
  only the *usable* set, so a target excluded by startup validation is reported
  as not-registered. ``exists`` consults *every* name present in the file so the
  duplicate guard cannot clobber an excluded entry.
- **Register (Req 10.1, 10.3-10.6).** :meth:`add` validates the candidate's name
  (via the model), rejects duplicate names without overwriting
  (:class:`~discord_ollama_agent.errors.DuplicateTargetError`), rejects paths
  that do not resolve inside the workspace
  (:class:`~discord_ollama_agent.errors.PathOutsideWorkspaceError`) and paths
  that do not refer to an existing directory
  (:class:`~discord_ollama_agent.errors.TargetDirectoryNotFoundError`), then
  persists the updated registry atomically (write-to-temp + ``os.replace``) so a
  crash mid-write can never corrupt the file (Req 10.1).

Workspace containment uses :meth:`pathlib.Path.resolve` on both the workspace
and the candidate, then checks the resolved candidate is relative to the
resolved workspace. Because ``resolve`` normalizes ``..`` and follows symbolic
links, this rejects parent-directory traversal and symlink escapes alike
(Req 10.3).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .errors import (
    DuplicateTargetError,
    PathOutsideWorkspaceError,
    StartupError,
    TargetDirectoryNotFoundError,
)
from .models.target import RegisteredTarget

__all__ = [
    "ExcludedTarget",
    "RegistryLoadResult",
    "TargetRegistry",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExcludedTarget:
    """A target excluded from the usable set during :meth:`TargetRegistry.load`.

    Attributes:
        name: The excluded target's name, or a positional placeholder (for
            example ``"<entry 3>"``) when the entry has no usable name.
        reason: Human-readable explanation naming the missing/offending value,
            suitable for the operator log (Req 12.2).
    """

    name: str
    reason: str


@dataclass(frozen=True)
class RegistryLoadResult:
    """The outcome of partitioning a loaded Target_Registry (Req 12.1-12.3).

    Attributes:
        usable: Targets that have all required per-target configuration and may
            be selected for a Job (Req 12.3).
        excluded: Targets that were dropped, each paired with the reason it was
            excluded (Req 12.2).
    """

    usable: list[RegisteredTarget] = field(default_factory=list)
    excluded: list[ExcludedTarget] = field(default_factory=list)


def _resolve_within(workspace_resolved: Path, candidate: Path) -> bool:
    """Return whether ``candidate`` resolves to a location inside the workspace.

    Resolves the candidate (normalizing ``..`` and following symlinks) and
    checks that the result is relative to the already-resolved workspace. Any
    resolution error (for example a symlink loop) is treated as "not contained"
    so a path that cannot be safely resolved is never accepted (Req 10.3).
    """
    try:
        resolved_candidate = candidate.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved_candidate.is_relative_to(workspace_resolved)


class TargetRegistry:
    """Persisted, in-workspace-confined store of Registered_Targets.

    A registry is populated by :meth:`load`; :meth:`add` mutates it and rewrites
    the backing file atomically. Until :meth:`load` has run, the registry is
    empty and :meth:`add` raises :class:`RuntimeError` because it has no backing
    path or workspace boundary to validate against.
    """

    def __init__(self) -> None:
        self._path: Path | None = None
        self._workspace_dir: Path | None = None
        self._workspace_resolved: Path | None = None
        # Usable targets, keyed by name, preserving load/registration order.
        self._usable: dict[str, RegisteredTarget] = {}
        # Every name present in the backing file (usable or excluded), so the
        # duplicate guard cannot clobber an excluded entry (Req 10.5).
        self._known_names: set[str] = set()
        # Raw, file-shaped entries (usable + excluded) so persistence preserves
        # excluded entries the operator may still want to repair.
        self._raw_entries: list[dict] = []

    def load(self, path: Path, workspace_dir: Path) -> RegistryLoadResult:
        """Load the persisted registry and partition it (Req 12.1-12.3).

        Reads the JSON registry at ``path`` and evaluates every entry against
        the required per-target configuration. Usable targets are retained for
        :meth:`get`/:meth:`names`; excluded targets are logged with their name
        and the missing/offending value and returned for inspection. A missing
        file is treated as an empty registry (so a fresh deployment can still
        accept ``/addtarget``).

        Args:
            path: Filesystem path to the JSON registry file.
            workspace_dir: The Workspace_Directory containment boundary.

        Returns:
            A :class:`RegistryLoadResult` with the usable and excluded targets.

        Raises:
            StartupError: If the file exists but is not valid JSON or its
                top-level shape is neither a list nor a ``{"targets": [...]}``
                mapping (the file as a whole cannot be partitioned).
        """
        self._path = Path(path)
        self._workspace_dir = Path(workspace_dir)
        self._workspace_resolved = self._workspace_dir.resolve()
        self._usable = {}
        self._known_names = set()
        self._raw_entries = []

        raw_entries = self._read_entries(self._path)
        result = RegistryLoadResult()

        for index, entry in enumerate(raw_entries):
            self._raw_entries.append(entry)
            excluded = self._classify_entry(index, entry, result)
            if excluded is not None:
                logger.error(
                    "Excluding target %r from the usable registry: %s",
                    excluded.name,
                    excluded.reason,
                )

        return result

    def _classify_entry(
        self, index: int, entry: object, result: RegistryLoadResult
    ) -> ExcludedTarget | None:
        """Sort a single raw registry entry into ``result``.

        Returns the :class:`ExcludedTarget` if the entry was excluded (so the
        caller can log it), or ``None`` if it was added to the usable set.
        """
        if not isinstance(entry, dict):
            excluded = ExcludedTarget(
                name=f"<entry {index}>",
                reason=f"registry entry is not a mapping (got {type(entry).__name__})",
            )
            result.excluded.append(excluded)
            return excluded

        raw_name = entry.get("name")
        display_name = raw_name if isinstance(raw_name, str) and raw_name.strip() else f"<entry {index}>"
        if isinstance(raw_name, str):
            self._known_names.add(raw_name)

        try:
            target = RegisteredTarget.model_validate(entry)
        except ValidationError as exc:
            excluded = ExcludedTarget(
                name=display_name,
                reason=f"missing or invalid configuration: {self._summarize_error(exc)}",
            )
            result.excluded.append(excluded)
            return excluded

        # The model guarantees a valid name; record it for the duplicate guard.
        self._known_names.add(target.name)

        if not target.repo_remote.strip():
            excluded = ExcludedTarget(
                name=target.name,
                reason="missing Target_Repository remote (repo_remote)",
            )
            result.excluded.append(excluded)
            return excluded

        assert self._workspace_resolved is not None  # set by load()
        if not _resolve_within(self._workspace_resolved, target.directory_path):
            excluded = ExcludedTarget(
                name=target.name,
                reason=(
                    f"directory_path {str(target.directory_path)!r} resolves outside "
                    f"the workspace {str(self._workspace_dir)!r}"
                ),
            )
            result.excluded.append(excluded)
            return excluded

        # A later usable entry with a duplicate name must not shadow an earlier
        # one; keep the first and exclude the rest.
        if target.name in self._usable:
            excluded = ExcludedTarget(
                name=target.name,
                reason="duplicate target name; first definition kept",
            )
            result.excluded.append(excluded)
            return excluded

        self._usable[target.name] = target
        result.usable.append(target)
        return None

    @staticmethod
    def _read_entries(path: Path) -> list:
        """Read and shape-check the registry file into a list of raw entries.

        Accepts either a top-level JSON list or a ``{"targets": [...]}`` object.
        A missing file yields an empty list. A malformed file raises
        :class:`StartupError`.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(
                "Target registry file not found at %s; starting with an empty registry",
                path,
            )
            return []
        except OSError as exc:
            raise StartupError(
                f"Target registry file could not be read: {path} ({exc})"
            ) from exc

        if not text.strip():
            return []

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StartupError(
                f"Target registry file is not valid JSON: {path} ({exc})"
            ) from exc

        if isinstance(data, dict):
            data = data.get("targets", [])
        if not isinstance(data, list):
            raise StartupError(
                "Target registry must be a JSON list of targets or an object with a "
                f"'targets' list; got {type(data).__name__} in {path}"
            )
        return data

    @staticmethod
    def _summarize_error(exc: ValidationError) -> str:
        """Render a pydantic ``ValidationError`` into a short, value-naming string."""
        parts = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ())) or "value"
            parts.append(f"{loc} ({error.get('msg', 'invalid value')})")
        return "; ".join(parts)

    def get(self, name: str) -> RegisteredTarget | None:
        """Return the usable Registered_Target named ``name``, or ``None``.

        Only the usable set is consulted, so a target excluded by startup
        validation is reported as absent and a ``/build`` against it is rejected
        as not-registered (Req 1.1, 12.4).
        """
        return self._usable.get(name)

    def names(self) -> list[str]:
        """Return the names of all usable Registered_Targets, in load order.

        The list reflects the usable set only and is empty when no usable
        targets exist (Req 9.1, 9.2).
        """
        return list(self._usable.keys())

    def exists(self, name: str) -> bool:
        """Return whether ``name`` is already present anywhere in the registry.

        This spans both usable and excluded entries so the registration
        duplicate guard cannot silently clobber an excluded target (Req 10.5).
        """
        return name in self._known_names

    def add(self, target: RegisteredTarget) -> None:
        """Register ``target`` and persist the registry atomically (Req 10.1).

        The target's name is already validated by the model (1..64 non-whitespace
        Unicode characters, Req 10.6). This method additionally enforces:

        - **Duplicate guard (Req 10.5).** A name already present in the registry
          is rejected without overwriting the existing entry.
        - **Workspace containment (Req 10.3).** The directory path must resolve
          inside the Workspace_Directory; ``..`` traversal and symlink escapes
          are rejected.
        - **Directory existence (Req 10.4).** The resolved path must refer to an
          existing directory.

        On success the updated registry is written to a temporary file in the
        registry's directory and atomically swapped into place with
        :func:`os.replace`, so an interrupted write cannot corrupt the registry.

        Args:
            target: The candidate Registered_Target to register.

        Raises:
            RuntimeError: If called before :meth:`load`.
            DuplicateTargetError: If the name already exists (Req 10.5).
            PathOutsideWorkspaceError: If the path resolves outside the
                workspace (Req 10.3).
            TargetDirectoryNotFoundError: If the path does not refer to an
                existing directory (Req 10.4).
        """
        if self._path is None or self._workspace_resolved is None:
            raise RuntimeError("TargetRegistry.add called before load()")

        if target.name in self._known_names:
            raise DuplicateTargetError(
                f"target name {target.name!r} is already in use; the existing target "
                "was not overwritten"
            )

        if not _resolve_within(self._workspace_resolved, target.directory_path):
            raise PathOutsideWorkspaceError(
                f"target path {str(target.directory_path)!r} must resolve inside the "
                f"workspace {str(self._workspace_dir)!r}"
            )

        resolved = target.directory_path.resolve()
        if not resolved.is_dir():
            raise TargetDirectoryNotFoundError(
                f"target path {str(target.directory_path)!r} does not refer to an "
                "existing directory"
            )

        entry = target.model_dump(mode="json")
        self._persist(self._raw_entries + [entry])

        # Only mutate in-memory state after the write has succeeded.
        self._raw_entries.append(entry)
        self._usable[target.name] = target
        self._known_names.add(target.name)

    def _persist(self, entries: list[dict]) -> None:
        """Atomically write ``entries`` to the backing file (write-temp + replace).

        The temporary file is created in the destination directory so that
        :func:`os.replace` is an atomic same-filesystem rename, and is fsync'd
        before the swap. A failed write leaves the original file untouched and
        removes the temporary file.
        """
        assert self._path is not None  # guarded by add()
        destination = self._path
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps({"targets": entries}, indent=2, ensure_ascii=False)

        fd, tmp_name = tempfile.mkstemp(
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, destination)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise
