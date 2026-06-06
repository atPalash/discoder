"""Integration tests for OllamaClient.generate timeout and error mapping (task 6.5).

These tests exercise the real :meth:`OllamaClient.generate` against a mocked
``httpx`` transport, so the request is constructed, dispatched, and the response
(or raised transport error) is mapped exactly as it would be in production --
only the network is replaced. They cover the two failure-mapping contracts the
Coding_Agent depends on to record the correct Job failure reason:

- A delayed/timed-out generation request maps to
  :class:`~discord_ollama_agent.errors.GenerationTimeout`, which the
  Coding_Agent records as a timeout reason (Req 3.10).
- An error response from the endpoint -- whether an HTTP error status or a
  logical ``error`` field in a 200 body -- maps to
  :class:`~discord_ollama_agent.errors.OllamaError`, which the Coding_Agent
  records as the returned error reason (Req 3.11).

A successful completion is included as a control so the failure cases are
attributable to the simulated error rather than to request construction.
"""

from __future__ import annotations

import httpx
import pytest

from discord_ollama_agent.errors import GenerationTimeout, OllamaError
from discord_ollama_agent.models.config import OllamaConfig
from discord_ollama_agent.models.generation import GenerationRequest
from discord_ollama_agent.ollama_client import OllamaClient


def _config() -> OllamaConfig:
    """A valid Ollama config pointing at a stub endpoint with a short timeout."""
    return OllamaConfig(
        endpoint_url="http://localhost:11434",
        model="llama3",
        generation_timeout_s=30,
    )


def _client(handler) -> OllamaClient:
    """Build an OllamaClient whose transport runs ``handler`` for each request."""
    transport = httpx.MockTransport(handler)
    return OllamaClient(_config(), client=httpx.AsyncClient(transport=transport))


# --- Timeout maps to GenerationTimeout (Req 3.10) --------------------------


async def test_generate_timeout_raises_generation_timeout() -> None:
    """A request that exceeds the generation timeout raises GenerationTimeout (Req 3.10).

    The mock transport raises ``httpx.TimeoutException`` exactly as the real
    transport would when no response arrives within the configured budget, so
    the Coding_Agent can record a timeout reason.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("simulated read timeout", request=request)

    client = _client(handler)
    try:
        with pytest.raises(GenerationTimeout):
            await client.generate(GenerationRequest(idea="add a feature"))
    finally:
        await client.aclose()


# --- Error response maps to OllamaError (Req 3.11) -------------------------


async def test_generate_error_status_raises_ollama_error() -> None:
    """A non-2xx response from the endpoint raises OllamaError (Req 3.11).

    The endpoint's error message is surfaced in the raised exception so the
    Coding_Agent can record the returned error reason.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model runner crashed"})

    client = _client(handler)
    try:
        with pytest.raises(OllamaError) as exc_info:
            await client.generate(GenerationRequest(idea="add a feature"))
    finally:
        await client.aclose()

    message = str(exc_info.value)
    assert "500" in message
    assert "model runner crashed" in message


async def test_generate_error_body_in_200_raises_ollama_error() -> None:
    """A 200 response carrying a logical ``error`` field raises OllamaError (Req 3.11).

    Ollama can report a logical failure inside an otherwise-successful HTTP
    response; this must still map to a recorded error reason.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "model not loaded"})

    client = _client(handler)
    try:
        with pytest.raises(OllamaError) as exc_info:
            await client.generate(GenerationRequest(idea="add a feature"))
    finally:
        await client.aclose()

    assert "model not loaded" in str(exc_info.value)


async def test_generate_transport_error_raises_ollama_error() -> None:
    """A non-timeout transport failure raises OllamaError (Req 3.11)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        with pytest.raises(OllamaError):
            await client.generate(GenerationRequest(idea="add a feature"))
    finally:
        await client.aclose()


# --- Control: a well-formed success does not raise --------------------------


async def test_generate_success_returns_completion_content() -> None:
    """A well-formed completion returns its content (control for the error cases)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"files": []}'}},
        )

    client = _client(handler)
    try:
        result = await client.generate(GenerationRequest(idea="add a feature"))
    finally:
        await client.aclose()

    assert result.content == '{"files": []}'
