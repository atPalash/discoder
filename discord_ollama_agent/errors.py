"""Shared exception hierarchy for the Discord-Ollama Coding Agent.

All custom exceptions raised across the system derive from :class:`AgentError`,
giving callers a single base type to catch when they want to handle any
agent-specific failure. The more specific exceptions identify distinct failure
domains referenced throughout the design:

- :class:`StartupError`      -- fail-fast system configuration/startup validation.
- :class:`SandboxInitError`  -- the per-Job sandbox working directory could not be created.
- :class:`PathEscapeError`   -- a path resolved outside the Job's sandbox boundary.
- :class:`WriteError`        -- writing a file entry into the sandbox failed.
- :class:`GenerationTimeout` -- the Ollama generation request exceeded its timeout.
- :class:`OllamaError`       -- the Ollama endpoint returned an error response.
- :class:`RegistrationError` -- a target could not be registered in the Target_Registry,
  with the distinct sub-causes :class:`DuplicateTargetError` (Req 10.5),
  :class:`PathOutsideWorkspaceError` (Req 10.3), and
  :class:`TargetDirectoryNotFoundError` (Req 10.4).
"""

__all__ = [
    "AgentError",
    "StartupError",
    "SandboxError",
    "SandboxInitError",
    "PathEscapeError",
    "WriteError",
    "OllamaError",
    "GenerationTimeout",
    "RegistrationError",
    "DuplicateTargetError",
    "PathOutsideWorkspaceError",
    "TargetDirectoryNotFoundError",
]


class AgentError(Exception):
    """Base class for all errors raised by the Discord-Ollama Coding Agent."""


class StartupError(AgentError):
    """Raised when fail-fast startup/configuration validation rejects a value.

    The message identifies the offending configuration value so the operator can
    correct it (for example, an out-of-bounds ``max_build_attempts`` or a missing
    Ollama endpoint).
    """


class SandboxError(AgentError):
    """Base class for failures originating in the Execution_Sandbox."""


class SandboxInitError(SandboxError):
    """Raised when a Job's sandbox working directory cannot be initialized."""


class PathEscapeError(SandboxError):
    """Raised when a path resolves outside the Job's sandbox working directory."""


class WriteError(SandboxError):
    """Raised when writing a file entry into the sandbox working directory fails."""


class OllamaError(AgentError):
    """Raised when the Ollama endpoint returns an error response."""


class GenerationTimeout(OllamaError):
    """Raised when an Ollama generation request exceeds its configured timeout."""


class RegistrationError(AgentError):
    """Base class for failures registering a target in the Target_Registry.

    The message identifies why the target was rejected so the Discord_Bot can
    surface a precise reply to the Admin_User (Req 10.3-10.6).
    """


class DuplicateTargetError(RegistrationError):
    """Raised when a target name already matches a Registered_Target (Req 10.5).

    The existing target is never overwritten.
    """


class PathOutsideWorkspaceError(RegistrationError):
    """Raised when a target path resolves outside the Workspace_Directory (Req 10.3).

    This covers parent-directory (``..``) traversal and symbolic links, because
    containment is checked on the fully resolved paths.
    """


class TargetDirectoryNotFoundError(RegistrationError):
    """Raised when a target path does not refer to an existing directory (Req 10.4)."""
