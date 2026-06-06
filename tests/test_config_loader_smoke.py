"""Smoke test for configuration loading (task 3.4).

A single happy-path execution confirming that a valid system-config YAML file
is read by the real :class:`ConfigLoader` and that the global Ollama endpoint
URL and model name (Req 2.1) together with the Discord Allowed_Channels
(Req 13.4) are populated on the resulting :class:`SystemConfig`.

This is a deterministic smoke test -- one read of one valid file -- and uses no
external services, matching the on-disk-YAML conventions in
``tests/test_config_loader_ollama.py``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from discord_ollama_agent.config_loader import ConfigLoader


def test_valid_config_populates_ollama_and_allowed_channels() -> None:
    """A valid config file loads and carries the Ollama endpoint/model and
    Allowed_Channels on the resulting ``SystemConfig`` (Req 2.1, 13.4)."""
    endpoint_url = "http://localhost:11434"
    model = "llama3"
    allowed_channels = [123456789, 987654321]

    config = {
        "ollama": {
            "endpoint_url": endpoint_url,
            "model": model,
        },
        "workspace_dir": "/tmp/agent-workspace",
        "max_build_attempts": 3,
        "concurrency_limit": 4,
        "context_budget": {
            "max_file_count": 10,
            "max_total_bytes": 4096,
        },
        "allowed_channels": allowed_channels,
    }

    loader = ConfigLoader()
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "system.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        result = loader.load(config_path)

    # Global Ollama endpoint URL and model are populated (Req 2.1). HttpUrl
    # normalises the URL, so compare on its string form by prefix.
    assert str(result.ollama.endpoint_url).rstrip("/") == endpoint_url
    assert result.ollama.model == model

    # Discord Allowed_Channels are populated (Req 13.4).
    assert result.allowed_channels == allowed_channels
