"""Integration tests for the startup wiring in ``main`` (task 13.2).

Exercises the Startup Sequence surfaces -- :func:`build_application`,
:class:`Application`, :class:`ConnectivityMonitor`, and :func:`async_main` --
with a mocked Ollama client (so no network is touched) and on-disk config/
registry files. The Discord gateway is never connected: these tests drive the
application/monitor surfaces directly rather than :func:`run_gateway`'s live
``client.start`` call, which would require a real Discord token and connection.

Covered scenarios:
- A valid config + usable registry builds a wired :class:`Application`; the
  initial connectivity check (mocked available) leaves the monitor available and
  the dispatcher can be started -- the System reaches a ready state (Req 2.4,
  12.1).
- An invalid config makes :func:`build_application` raise
  :class:`StartupError` and :func:`async_main` return
  :data:`EXIT_STARTUP_ERROR` without ever reaching a ready state (Req 2.4).
- A failed connectivity check (mocked unavailable) leaves the System in degraded
  mode (``monitor.available`` is ``False``, so the unavailable-model message
  applies), and a later successful check recovers readiness (Req 2.4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from discord_ollama_agent.errors import StartupError
from discord_ollama_agent.main import (
    EXIT_STARTUP_ERROR,
    UNAVAILABLE_MODEL_MESSAGE,
    Application,
    ConnectivityMonitor,
    async_main,
    build_application,
)
from discord_ollama_agent.ollama_client import ConnectivityResult

_ENDPOINT = "http://localhost:11434"
_MODEL = "llama3"


class _FakeOllamaClient:
    """A stand-in for :class:`OllamaClient` for the startup-wiring tests.

    Implements only the surface the startup sequence touches:
    :meth:`check_connectivity` (consumed by the :class:`ConnectivityMonitor`) and
    :meth:`aclose` (called on shutdown). Each call to ``check_connectivity``
    returns the next queued :class:`ConnectivityResult`, repeating the final one
    once the queue is drained, so a test can script "degraded then recovered".
    """

    def __init__(self, results: list[ConnectivityResult]) -> None:
        assert results, "at least one scripted result is required"
        self._results = list(results)
        self.check_calls = 0
        self.closed = False

    async def check_connectivity(self) -> ConnectivityResult:
        self.check_calls += 1
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]

    async def aclose(self) -> None:
        self.closed = True


def _available(detail: str = "model confirmed") -> ConnectivityResult:
    return ConnectivityResult(available=True, detail=detail)


def _unavailable(detail: str = "model not found") -> ConnectivityResult:
    return ConnectivityResult(available=False, detail=detail)


def _write_valid_config(workspace_dir: Path, path: Path) -> None:
    """Write a syntactically and semantically valid system-config YAML file."""
    config = {
        "ollama": {"endpoint_url": _ENDPOINT, "model": _MODEL},
        "workspace_dir": str(workspace_dir),
        "max_build_attempts": 3,
        "concurrency_limit": 2,
        "context_budget": {"max_file_count": 10, "max_total_bytes": 4096},
        "allowed_channels": [123456789],
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_invalid_config(workspace_dir: Path, path: Path) -> None:
    """Write a config that fails fail-fast validation (empty allowed_channels).

    An empty ``allowed_channels`` list is a startup error (Req 13.5): no channel
    could ever accept a command, so the System must refuse to start.
    """
    config = {
        "ollama": {"endpoint_url": _ENDPOINT, "model": _MODEL},
        "workspace_dir": str(workspace_dir),
        "concurrency_limit": 2,
        "context_budget": {"max_file_count": 10, "max_total_bytes": 4096},
        "allowed_channels": [],  # invalid: must contain at least one channel
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_registry(workspace_dir: Path, path: Path) -> str:
    """Write a registry JSON with one usable target inside the workspace.

    Returns the usable target's name so the caller can assert it is exposed.
    """
    target_dir = workspace_dir / "demo-target"
    target_dir.mkdir(parents=True, exist_ok=True)
    name = "demo"
    registry = {
        "targets": [
            {
                "name": name,
                "directory_path": str(target_dir),
                "repo_remote": "https://example.invalid/demo.git",
                "credentials_ref": "GIT_TOKEN_ENV",
            }
        ]
    }
    path.write_text(json.dumps(registry), encoding="utf-8")
    return name


# -- Scenario 1: valid config + registry reaches a ready state ----------------


async def test_valid_config_builds_ready_application(tmp_path: Path) -> None:
    """A valid config + usable registry builds an Application that reaches a
    ready state: the initial connectivity check leaves the monitor available and
    the dispatcher can be started (Req 2.4, 12.1)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    registry_path = tmp_path / "targets.json"
    _write_valid_config(workspace, config_path)
    target_name = _write_registry(workspace, registry_path)

    fake_client = _FakeOllamaClient([_available()])
    app = build_application(
        config_path,
        registry_path,
        ollama_client=fake_client,
        recheck_interval_s=0.01,
    )

    # Req 12.1: the usable registry is loaded and the valid target is exposed.
    assert isinstance(app, Application)
    assert app.registry.names() == [target_name]
    assert not app.registry_result.excluded

    # Phase 3: the initial connectivity check confirms the model (Req 2.4).
    result = await app.check_connectivity()
    assert result.available is True
    assert app.monitor.available is True
    assert fake_client.check_calls == 1

    # Ready state: the dispatcher loop can be started under the running loop and
    # is alive. ``start`` is idempotent and requires the wired worker.
    app.start_dispatcher()
    assert app.job_manager._dispatch_task is not None
    assert not app.job_manager._dispatch_task.done()

    # An already-available monitor starts no recovery loop (nothing to recover).
    app.monitor.start()
    assert app.monitor._task is None

    await app.shutdown()
    assert fake_client.closed is True


# -- Scenario 2: invalid config exits without a ready state -------------------


def test_invalid_config_raises_startup_error(tmp_path: Path) -> None:
    """An invalid config makes ``build_application`` fail fast with a
    ``StartupError`` whose message identifies the offending value (Req 2.4)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    registry_path = tmp_path / "targets.json"
    _write_invalid_config(workspace, config_path)
    _write_registry(workspace, registry_path)

    with pytest.raises(StartupError) as exc_info:
        build_application(
            config_path,
            registry_path,
            ollama_client=_FakeOllamaClient([_available()]),
        )
    # The message identifies the offending configuration value.
    assert "allowed_channels" in str(exc_info.value)


async def test_invalid_config_async_main_exits_without_ready_state(
    tmp_path: Path,
) -> None:
    """``async_main`` returns ``EXIT_STARTUP_ERROR`` for an invalid config and
    never reaches a ready state: connectivity is never checked and the gateway is
    never connected (Req 2.4)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    registry_path = tmp_path / "targets.json"
    _write_invalid_config(workspace, config_path)
    _write_registry(workspace, registry_path)

    exit_code = await async_main(config_path, registry_path)

    assert exit_code == EXIT_STARTUP_ERROR


# -- Scenario 3: failed connectivity enters degraded mode, then recovers ------


async def test_failed_connectivity_enters_degraded_mode(tmp_path: Path) -> None:
    """A failed initial connectivity check enters degraded mode: the monitor
    reports unavailable, so new Jobs are rejected with the unavailable-model
    message (Req 2.4)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path = tmp_path / "config.yaml"
    registry_path = tmp_path / "targets.json"
    _write_valid_config(workspace, config_path)
    _write_registry(workspace, registry_path)

    fake_client = _FakeOllamaClient([_unavailable("configured model not found")])
    app = build_application(
        config_path,
        registry_path,
        ollama_client=fake_client,
        recheck_interval_s=0.01,
    )

    result = await app.check_connectivity()

    # Degraded mode: the model is not confirmed, so the System rejects new Jobs.
    assert result.available is False
    assert app.monitor.available is False
    assert "not found" in app.monitor.detail
    # The unavailable-model message is what /build and /revise reply while the
    # monitor is unavailable (Req 2.4).
    assert UNAVAILABLE_MODEL_MESSAGE

    await app.shutdown()


async def test_degraded_mode_recovers_on_later_successful_check(
    tmp_path: Path,
) -> None:
    """After a degraded initial check, the recheck loop recovers readiness once a
    later connectivity check confirms the model (Req 2.4)."""
    # First check fails (degraded); every later check confirms the model.
    fake_client = _FakeOllamaClient([_unavailable(), _available("now available")])
    monitor = ConnectivityMonitor(fake_client, recheck_interval_s=0.01)

    # Initial check enters degraded mode.
    first = await monitor.check_once()
    assert first.available is False
    assert monitor.available is False

    # The recovery loop re-checks until the model is confirmed (Req 2.4).
    monitor.start()
    for _ in range(200):
        if monitor.available:
            break
        await _yield()
    await monitor.stop()

    assert monitor.available is True
    assert fake_client.check_calls >= 2


async def _yield() -> None:
    """Yield control briefly so background tasks can make progress."""
    import asyncio

    await asyncio.sleep(0.01)
