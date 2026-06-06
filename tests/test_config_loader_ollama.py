"""Example/edge tests for Ollama configuration validation (task 3.3).

Deterministic pytest examples covering the fail-fast Ollama startup checks the
``ConfigLoader`` performs when constructing a ``SystemConfig``:

- A missing or empty Ollama endpoint URL raises :class:`StartupError`  (Req 2.2)
- A missing or empty Ollama model name raises :class:`StartupError`    (Req 2.2)
- An endpoint URL present but not a valid http/https URL raises
  :class:`StartupError`                                                (Req 2.5)

Each raised ``StartupError`` must carry a message identifying the offending
configuration value (``ollama.endpoint_url`` or ``ollama.model``). The tests
exercise the real loader against on-disk YAML and use no external services,
matching the conventions in ``tests/test_config_loader.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from discord_ollama_agent.config_loader import ConfigLoader
from discord_ollama_agent.errors import StartupError

# Message fragments the loader uses to identify each offending Ollama value.
_ENDPOINT_LABEL = "ollama.endpoint_url"
_MODEL_LABEL = "ollama.model"


def _valid_config() -> dict:
    """Return a fully valid system-config mapping.

    Every required value is in bounds so that any raised ``StartupError`` is
    attributable solely to the Ollama field a test deliberately corrupts.
    """
    return {
        "ollama": {
            "endpoint_url": "http://localhost:11434",
            "model": "llama3",
        },
        "workspace_dir": "/tmp/agent-workspace",
        "max_build_attempts": 3,
        "concurrency_limit": 4,
        "context_budget": {
            "max_file_count": 10,
            "max_total_bytes": 4096,
        },
        "allowed_channels": [123456789],
    }


def _load_config(config: dict) -> None:
    """Write ``config`` to a temp YAML file and load it via ``ConfigLoader``.

    Raises whatever ``ConfigLoader.load`` raises (a ``StartupError`` for the
    invalid configurations under test).
    """
    loader = ConfigLoader()
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "system.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loader.load(config_path)


def test_valid_ollama_config_loads() -> None:
    """A fully valid Ollama config loads without error (control case)."""
    # Sanity check: the baseline used by the negative cases is itself valid, so
    # a failure below is caused only by the deliberately corrupted value.
    _load_config(_valid_config())


# --- Missing / empty Ollama endpoint URL (Req 2.2) -------------------------


def test_missing_endpoint_url_raises_startup_error() -> None:
    """An absent Ollama endpoint URL terminates startup (Req 2.2)."""
    config = _valid_config()
    del config["ollama"]["endpoint_url"]

    with pytest.raises(StartupError) as exc_info:
        _load_config(config)

    assert _ENDPOINT_LABEL in str(exc_info.value)


def test_empty_endpoint_url_raises_startup_error() -> None:
    """An empty-string Ollama endpoint URL terminates startup (Req 2.2)."""
    config = _valid_config()
    config["ollama"]["endpoint_url"] = ""

    with pytest.raises(StartupError) as exc_info:
        _load_config(config)

    assert _ENDPOINT_LABEL in str(exc_info.value)


# --- Missing / empty Ollama model name (Req 2.2) ---------------------------


def test_missing_model_raises_startup_error() -> None:
    """An absent Ollama model name terminates startup (Req 2.2)."""
    config = _valid_config()
    del config["ollama"]["model"]

    with pytest.raises(StartupError) as exc_info:
        _load_config(config)

    assert _MODEL_LABEL in str(exc_info.value)


def test_empty_model_raises_startup_error() -> None:
    """An empty-string Ollama model name terminates startup (Req 2.2)."""
    config = _valid_config()
    config["ollama"]["model"] = ""

    with pytest.raises(StartupError) as exc_info:
        _load_config(config)

    assert _MODEL_LABEL in str(exc_info.value)


def test_whitespace_only_model_raises_startup_error() -> None:
    """A whitespace-only Ollama model name is treated as empty (Req 2.2)."""
    config = _valid_config()
    config["ollama"]["model"] = "   \t  "

    with pytest.raises(StartupError) as exc_info:
        _load_config(config)

    assert _MODEL_LABEL in str(exc_info.value)


# --- Endpoint present but not a valid http/https URL (Req 2.5) -------------


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "ftp://localhost:11434",  # wrong scheme
        "ws://localhost:11434",   # wrong scheme
        "localhost:11434",        # missing scheme
        "not a url",              # not a URL at all
        "://missing-scheme",      # malformed
    ],
)
def test_non_http_endpoint_url_raises_startup_error(endpoint_url: str) -> None:
    """A present-but-invalid (non-http/https) endpoint URL terminates startup
    (Req 2.5), and the error identifies the offending endpoint value."""
    config = _valid_config()
    config["ollama"]["endpoint_url"] = endpoint_url

    with pytest.raises(StartupError) as exc_info:
        _load_config(config)

    assert _ENDPOINT_LABEL in str(exc_info.value)
