"""Config_Loader: fail-fast loading and validation of system configuration.

Reads the system configuration (YAML) and constructs a validated
:class:`~discord_ollama_agent.models.config.SystemConfig`, translating every
validation failure into a :class:`~discord_ollama_agent.errors.StartupError` so
that misconfiguration prevents the System from entering its ready state.

Validation is fail-fast: a missing file, malformed YAML, a non-mapping document,
or any pydantic bound violation raises a ``StartupError`` whose message
identifies the offending value -- for example the Ollama endpoint/model
(Req 2.2), the endpoint URL scheme (Req 2.5), ``max_build_attempts`` (Req 4.9),
``concurrency_limit`` (Req 7.6), the context-budget file count / byte size
(Req 4.14), or an empty ``allowed_channels`` list (Req 13.5). On success the
loaded settings (including the Ollama endpoint/model and Allowed_Channels) are
populated on the returned ``SystemConfig`` (Req 2.1, 13.4).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .errors import StartupError
from .models.config import SystemConfig

__all__ = ["ConfigLoader"]


#: Human-readable labels for the configuration fields, keyed by the string parts
#: of a pydantic error location. Used to turn an internal field path into a
#: message that identifies the offending value for the operator.
_FIELD_LABELS: dict[tuple[str, ...], str] = {
    ("ollama",): "Ollama configuration (ollama)",
    ("ollama", "endpoint_url"): "Ollama endpoint URL (ollama.endpoint_url)",
    ("ollama", "model"): "Ollama model name (ollama.model)",
    ("ollama", "generation_timeout_s"): (
        "Ollama generation timeout (ollama.generation_timeout_s)"
    ),
    ("workspace_dir",): "workspace directory (workspace_dir)",
    ("max_build_attempts",): "maximum build attempts (max_build_attempts)",
    ("concurrency_limit",): "concurrency limit (concurrency_limit)",
    ("execution_timeout_s",): "execution timeout (execution_timeout_s)",
    ("context_budget",): "context budget (context_budget)",
    ("context_budget", "max_file_count"): (
        "context budget maximum file count (context_budget.max_file_count)"
    ),
    ("context_budget", "max_total_bytes"): (
        "context budget maximum total size in bytes (context_budget.max_total_bytes)"
    ),
    ("allowed_channels",): "allowed channels (allowed_channels)",
    ("authorized_users",): "authorized users (authorized_users)",
    ("admin_users",): "admin users (admin_users)",
    ("default_target",): "default target (default_target)",
    ("max_idea_length",): "maximum idea length (max_idea_length)",
    ("push_timeout_s",): "push timeout (push_timeout_s)",
    ("build_output_cap_bytes",): "build output cap in bytes (build_output_cap_bytes)",
}


def _describe_location(loc: tuple[object, ...]) -> str:
    """Return a human-readable description for a pydantic error location.

    Looks up the string parts of ``loc`` in :data:`_FIELD_LABELS`, falling back
    to a dotted join of the parts. A trailing list index (for example a bad
    element of ``allowed_channels``) is appended so the offending element is
    identified precisely.
    """
    string_parts = tuple(part for part in loc if isinstance(part, str))
    label = _FIELD_LABELS.get(string_parts)
    if label is None:
        label = ".".join(string_parts) if string_parts else "configuration value"

    index = next((part for part in loc if isinstance(part, int)), None)
    if index is not None:
        label = f"{label} (item {index})"
    return label


class ConfigLoader:
    """Loads and fail-fast-validates the system configuration from YAML."""

    def load(self, path: str | Path) -> SystemConfig:
        """Read ``path`` as YAML and construct a validated ``SystemConfig``.

        Args:
            path: Filesystem path to the system configuration YAML file.

        Returns:
            The validated :class:`SystemConfig`.

        Raises:
            StartupError: If the file is missing or unreadable, the content is
                not valid YAML, the document is not a mapping, or any value
                fails validation. The message identifies the offending value.
        """
        config_path = Path(path)

        try:
            raw = config_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise StartupError(
                f"System configuration file not found: {config_path}"
            ) from exc
        except OSError as exc:
            raise StartupError(
                f"System configuration file could not be read: {config_path} ({exc})"
            ) from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise StartupError(
                f"System configuration file is not valid YAML: {config_path} ({exc})"
            ) from exc

        if data is None:
            raise StartupError(
                f"System configuration file is empty: {config_path}"
            )
        if not isinstance(data, dict):
            raise StartupError(
                "System configuration must be a mapping of settings; got "
                f"{type(data).__name__}"
            )

        try:
            return SystemConfig.model_validate(data)
        except ValidationError as exc:
            raise StartupError(self._format_validation_error(exc)) from exc

    @staticmethod
    def _format_validation_error(exc: ValidationError) -> str:
        """Translate a pydantic ``ValidationError`` into a ``StartupError`` message.

        Each underlying error is rendered as ``<offending value>: <reason>`` so
        the combined message names every value that failed validation.
        """
        details = [
            f"{_describe_location(error['loc'])}: {error.get('msg', 'invalid value')}"
            for error in exc.errors()
        ]
        return "Invalid system configuration: " + "; ".join(details)
