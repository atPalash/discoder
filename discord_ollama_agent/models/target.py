"""Registered target data model for the Discord-Ollama Coding Agent.

A :class:`RegisteredTarget` is a named target directory the System knows about,
carrying its own per-target configuration: a unique name, a directory path that
must resolve inside the Workspace_Directory, an optional Build_Command, its own
Git remote, its own branch-naming scheme, and a *reference* to its Git
credentials (Req 5.1, 5.3, 5.4, 5.7, 10.6, 12.1).

Bounds are enforced at construction so that registry loading and registration
fail fast with a precise message identifying the offending value. The name must
contain at least one non-whitespace character and be at most 64 Unicode
characters (Req 10.6, Property 26).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator

__all__ = [
    "RegisteredTarget",
    "TARGET_NAME_MAX_LENGTH",
]

#: Maximum allowed target-name length, measured in Unicode characters (Req 10.6).
TARGET_NAME_MAX_LENGTH = 64


class RegisteredTarget(BaseModel):
    """A single target the agent can build against.

    Fields:
        name: Unique target name. Must contain at least one non-whitespace
            character and be at most 64 Unicode characters (Req 10.6).
        directory_path: Target directory; must resolve inside the
            Workspace_Directory. Containment is enforced by the Target_Registry
            at registration/load time, not here (Req 10.1, 10.3).
        build_command: Optional command run to build/test generated changes. A
            target with no build command skips the build step (Req 4.3, 4.7).
        repo_remote: The target's Git remote URL; required for usability so the
            agent can push produced branches (Req 5.3, 12.1).
        default_branch: Branch the working copy is synced to before each Job
            (Req 4.12).
        branch_scheme: Branch-naming template. The default ``"{job_id}"`` yields
            a branch name that includes the target name, because the Job id is
            ``"{target_name}-{unique_suffix}"`` (Req 5.1).
        credentials_ref: A *reference* to the target's Git credentials -- an
            environment-variable name or a secret path -- never the secret value
            itself. The actual secret is resolved at push time and is never
            persisted to the registry or written to logs (Req 5.4, 5.7).
    """

    name: str
    directory_path: Path
    build_command: str | None = None
    repo_remote: str
    default_branch: str = "main"
    branch_scheme: str = "{job_id}"
    # credentials_ref holds a REFERENCE to the credentials (env var name or
    # secret path), NOT the secret value. The secret is resolved at runtime and
    # never stored or logged (Req 5.4, 5.7).
    credentials_ref: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Enforce the target-name bounds (Req 10.6, Property 26).

        A name is accepted iff it has at least one non-whitespace character and
        its length measured in Unicode characters is at most
        :data:`TARGET_NAME_MAX_LENGTH`. Whitespace-only or over-length names are
        rejected with a message identifying the offending value.
        """
        if not value.strip():
            raise ValueError(
                "target name must contain at least one non-whitespace character; "
                f"got {value!r}"
            )
        if len(value) > TARGET_NAME_MAX_LENGTH:
            raise ValueError(
                f"target name must be at most {TARGET_NAME_MAX_LENGTH} Unicode "
                f"characters; got {len(value)} characters: {value!r}"
            )
        return value
