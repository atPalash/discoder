"""Generation data models for the Discord-Ollama Coding Agent.

These pydantic v2 models describe the structured contract between the agent and
the locally hosted Ollama model:

- :class:`FileEntry`        -- a single whole-file write (relative path + full content).
- :class:`GenerationResult` -- the model's output: zero or more file entries.
- :class:`GenerationRequest`-- the input handed to the model: the idea plus any
  budget-bounded existing-file context, a prior error for retries, and feedback
  for revisions.

:class:`GenerationResult` is serialized to JSON Schema via
``GenerationResult.model_json_schema()`` and passed as Ollama's structured-output
``format`` parameter, so the models must round-trip cleanly through
``model_dump()`` / ``model_validate()`` (Req 3.3, 3.7).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "FileEntry",
    "GenerationResult",
    "GenerationRequest",
]


class FileEntry(BaseModel):
    """A single file the model intends to write into the Job sandbox.

    ``path`` is interpreted as a path relative to the Job's working directory and
    is validated against the sandbox boundary at write time (Req 3.3, 4.2).
    ``content`` is the full intended contents of the file -- writes are
    whole-file replacements rather than diffs/patches (Req 3.3, 4.1).
    """

    path: str
    content: str


class GenerationResult(BaseModel):
    """The structured output produced by a single generation attempt.

    ``files`` may contain zero or more entries. A result with zero entries is
    treated as an empty-generation failure by the Coding_Agent (Req 3.7, 3.12).
    Its JSON schema is used as Ollama's structured-output ``format`` parameter.
    """

    files: list[FileEntry] = Field(default_factory=list)


class GenerationRequest(BaseModel):
    """The input assembled for a single generation attempt.

    ``context_files`` carries existing working-copy files within the configured
    context budget, or is empty when the working copy is empty (Req 3.1, 3.2,
    3.4, 3.5). ``prior_error`` feeds a previous parse error or build output back
    into a retry (Req 3.8, 4.5). ``feedback`` carries revision instructions
    (Req 11.2).
    """

    idea: str
    context_files: list[FileEntry] = Field(default_factory=list)
    prior_error: str | None = None
    feedback: str | None = None
