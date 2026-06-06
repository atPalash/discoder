"""Integration tests for ``OllamaClient.check_connectivity`` (task 6.4).

Exercises the startup connectivity check end-to-end against a mocked Ollama
``GET /api/tags`` endpoint, using an injected :class:`httpx.AsyncClient` backed
by an :class:`httpx.MockTransport`. No real network or Ollama server is touched.

The ``available`` flag of the returned :class:`ConnectivityResult` is exactly
what drives the System's degraded-but-running mode: ``True`` means the model is
confirmed and Jobs may run; ``False`` means the System logs the reason and
rejects new Jobs with an unavailable-model message until a later check
confirms the model (Req 2.3, 2.4).

Covered scenarios:
- Model present in the tag list -> ready (``available`` is ``True``)   (Req 2.3)
- Model absent from the tag list -> degraded (``available`` is ``False``) (Req 2.4)
- Endpoint unreachable -> degraded                                       (Req 2.4)
- Endpoint times out within the 10s budget -> degraded                   (Req 2.4)
- A later successful check confirms the model and recovers readiness     (Req 2.4)
"""

from __future__ import annotations

import httpx

from discord_ollama_agent.models.config import OllamaConfig
from discord_ollama_agent.ollama_client import OllamaClient

_ENDPOINT = "http://localhost:11434"
_MODEL = "llama3"


def _make_client(
    handler: "callable[[httpx.Request], httpx.Response]",
    *,
    model: str = _MODEL,
) -> OllamaClient:
    """Build an ``OllamaClient`` whose HTTP traffic is served by ``handler``.

    The injected :class:`httpx.AsyncClient` uses a :class:`httpx.MockTransport`,
    so every request the client makes is answered by ``handler`` instead of the
    network.
    """
    config = OllamaConfig(endpoint_url=_ENDPOINT, model=model)
    transport = httpx.MockTransport(handler)
    async_client = httpx.AsyncClient(transport=transport)
    return OllamaClient(config, client=async_client)


def _tags_response(*names: str) -> httpx.Response:
    """Return a 200 ``/api/tags`` payload listing the given model ``names``."""
    return httpx.Response(
        200, json={"models": [{"name": name} for name in names]}
    )


async def test_model_present_confirms_readiness() -> None:
    """A tag list containing the configured model reports availability (Req 2.3)."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return _tags_response("llama3:latest", "mistral:latest")

    client = _make_client(handler)
    try:
        result = await client.check_connectivity()
    finally:
        await client.aclose()

    assert result.available is True
    assert _MODEL in result.detail
    # The check probes exactly the /api/tags endpoint.
    assert requested == ["/api/tags"]


async def test_model_absent_flips_degraded_mode() -> None:
    """A tag list lacking the configured model reports unavailability (Req 2.4)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _tags_response("mistral:latest", "phi3:latest")

    client = _make_client(handler)
    try:
        result = await client.check_connectivity()
    finally:
        await client.aclose()

    assert result.available is False
    # The failure reason names the missing model so it can be logged (Req 2.4).
    assert _MODEL in result.detail


async def test_unreachable_endpoint_flips_degraded_mode() -> None:
    """An unreachable endpoint reports unavailability without raising (Req 2.4)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _make_client(handler)
    try:
        result = await client.check_connectivity()
    finally:
        await client.aclose()

    assert result.available is False
    assert _ENDPOINT in result.detail


async def test_timeout_within_budget_flips_degraded_mode() -> None:
    """A check that times out within the 10s budget reports unavailability (Req 2.4)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = _make_client(handler)
    try:
        result = await client.check_connectivity()
    finally:
        await client.aclose()

    assert result.available is False
    assert "timed out" in result.detail.lower()


async def test_later_check_recovers_readiness() -> None:
    """After an initial failure, a subsequent check confirming the model
    recovers readiness (Req 2.4).

    The first probe fails (model absent -> degraded); the second probe, once the
    model has been pulled, lists it and confirms availability again.
    """
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            # First check: model not yet available -> degraded mode.
            return _tags_response("mistral:latest")
        # Later check: the model is now present -> readiness recovers.
        return _tags_response("llama3:latest")

    client = _make_client(handler)
    try:
        first = await client.check_connectivity()
        second = await client.check_connectivity()
    finally:
        await client.aclose()

    assert first.available is False
    assert second.available is True
    assert _MODEL in second.detail
