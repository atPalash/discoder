"""System configuration data models for the Discord-Ollama Coding Agent.

These pydantic v2 models describe the global, operator-authored configuration
the System reads at startup. All validation bounds are enforced **at
construction** so that startup fails fast with a precise message identifying the
offending value (the Config_Loader, implemented separately, translates these
validation errors into a ``StartupError``):

- :class:`ContextBudget` -- caps the existing-project context handed to the model
  (max file count and max total bytes) (Req 4.14).
- :class:`OllamaConfig`  -- the global Ollama endpoint URL, model name, and the
  per-request generation timeout (Req 2.1, 2.2, 2.5, 3.10).
- :class:`SystemConfig`  -- the top-level configuration aggregating the Ollama
  settings, workspace location, concurrency/attempt bounds, the context budget,
  and the Discord allowlists/channels (Req 4.6, 4.8, 4.9, 5.5, 7.6, 13.4, 13.5).

Only the ``Ollama_Endpoint`` URL and model are global across all targets; every
other per-target setting lives in the Target_Registry, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, field_validator

__all__ = [
    "ContextBudget",
    "OllamaConfig",
    "SystemConfig",
    "DEFAULT_BUILD_OUTPUT_CAP_BYTES",
]

#: Default cap on captured build/test output, 1 MiB (Req 4.6).
DEFAULT_BUILD_OUTPUT_CAP_BYTES = 1_048_576


class ContextBudget(BaseModel):
    """Bounds on the existing-project context included in a generation request.

    The budget keeps a generation request within the model's context window by
    capping both how many existing files and how many total bytes are supplied
    as context (Req 4.14, Property 9). Both bounds are enforced at construction;
    an absent, non-integer, or out-of-range value raises a validation error that
    identifies the offending value.

    Fields:
        max_file_count: Maximum number of existing files included as context.
            Integer in [1, 1000] (Req 4.14).
        max_total_bytes: Maximum combined size, in bytes, of the included
            context files. Integer in [1024, 67108864] (Req 4.14).
    """

    max_file_count: Annotated[int, Field(ge=1, le=1000)]
    max_total_bytes: Annotated[int, Field(ge=1024, le=67_108_864)]


class OllamaConfig(BaseModel):
    """Global configuration for the locally hosted Ollama server.

    These are the only settings shared across every Registered_Target. The
    endpoint URL must be a syntactically valid http/https URL (``HttpUrl``
    rejects other schemes), the model name must contain at least one
    non-whitespace character, and the per-request generation timeout is bounded
    (Req 2.1, 2.2, 2.5, 3.10). Each is validated at construction so a missing,
    empty, or invalid value raises a message identifying the offending value.

    Fields:
        endpoint_url: The Ollama server URL. Must be a valid http or https URL
            (Req 2.1, 2.5).
        model: The model name to generate with. Must be non-empty /
            non-whitespace (Req 2.1, 2.2).
        generation_timeout_s: Per-request generation timeout in seconds. Integer
            in [30, 600], default 120 (Req 3.10).
    """

    endpoint_url: HttpUrl
    model: str
    generation_timeout_s: Annotated[int, Field(ge=30, le=600)] = 120

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        """Reject an empty or whitespace-only model name (Req 2.2).

        A blank model name is a missing-configuration error; the message
        identifies the offending value so the Config_Loader can surface it.
        """
        if not value.strip():
            raise ValueError(
                "Ollama model name must contain at least one non-whitespace "
                f"character; got {value!r}"
            )
        return value


class SystemConfig(BaseModel):
    """Top-level system configuration validated fail-fast at startup.

    Aggregates the global Ollama settings, the workspace location, the
    concurrency/attempt/timeout bounds, the context budget, and the Discord
    allowlists and channels. Every numeric bound is enforced at construction so
    that an out-of-bounds, non-integer, or missing required value raises a
    validation error whose message identifies the offending value (Req 4.9,
    4.14, 7.6, 13.5, Property 8).

    Fields:
        ollama: Global Ollama endpoint/model/timeout settings.
        workspace_dir: Parent folder under which all valid target directories
            must reside (the containment boundary used by the Target_Registry).
        max_build_attempts: Maximum generate-and-build attempts per Job. Integer
            in [1, 10], default 3 (Req 4.9).
        concurrency_limit: Maximum number of concurrently Running Jobs. Integer
            in [1, 100] (Req 7.6).
        execution_timeout_s: Maximum build/test command runtime in seconds.
            Integer in [1, 3600], default 300 (Req 4.8).
        context_budget: Bounds on existing-project context (see
            :class:`ContextBudget`) (Req 4.14).
        allowed_channels: Discord channel identifiers in which commands are
            accepted. Must be non-empty (Req 13.4, 13.5).
        authorized_users: Discord user identifiers permitted to run
            build/status/cancel/targets/revise.
        admin_users: Discord user identifiers permitted to run ``/addtarget``
            (Req 10.2).
        default_target: Registered_Target used by ``/build`` when no target is
            supplied; optional (Req 1.2, 1.7).
        max_idea_length: Optional cap on idea length in Unicode characters; when
            set it must be a positive integer (Req 1.8).
        push_timeout_s: Maximum time, in seconds, allowed for a push. Integer in
            [1, 3600], default 120 (Req 5.5).
        build_output_cap_bytes: Maximum captured build output, in bytes. Integer
            >= 1, default 1 MiB (Req 4.6).
    """

    ollama: OllamaConfig
    workspace_dir: Path
    max_build_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    concurrency_limit: Annotated[int, Field(ge=1, le=100)]
    execution_timeout_s: Annotated[int, Field(ge=1, le=3600)] = 300
    context_budget: ContextBudget
    allowed_channels: list[int]
    authorized_users: list[int] = Field(default_factory=list)
    admin_users: list[int] = Field(default_factory=list)
    default_target: str | None = None
    max_idea_length: Annotated[int, Field(ge=1)] | None = None
    push_timeout_s: Annotated[int, Field(ge=1, le=3600)] = 120
    build_output_cap_bytes: Annotated[int, Field(ge=1)] = DEFAULT_BUILD_OUTPUT_CAP_BYTES

    @field_validator("allowed_channels")
    @classmethod
    def _validate_allowed_channels(cls, value: list[int]) -> list[int]:
        """Require at least one Allowed_Channel (Req 13.5, Property 8).

        An empty list means no channel could ever accept a command, so it is a
        startup error; the message identifies the offending value.
        """
        if not value:
            raise ValueError(
                "allowed_channels must contain at least one channel identifier; "
                "got an empty list"
            )
        return value
