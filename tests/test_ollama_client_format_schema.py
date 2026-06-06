"""Example test for the structured-output ``format`` field (task 6.3).

Verifies that :meth:`OllamaClient.generate` issues a ``POST /api/chat`` request
whose ``format`` field carries the JSON schema derived from
:class:`~discord_ollama_agent.models.generation.GenerationResult`
(``model_json_schema()``), which is what constrains the model to emit
schema-shaped JSON (Req 3.3).
"""

from __future__ import annotations

import json

import httpx
import pytest

from discord_ollama_agent.models.config import OllamaConfig
from discord_ollama_agent.models.generation import GenerationRequest, GenerationResult
from discord_ollama_agent.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_generate_request_carries_generation_result_schema_in_format() -> None:
    """The generation POST body's ``format`` equals ``GenerationResult.model_json_schema()``."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Record the outgoing request body so we can inspect the ``format`` field.
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "{\"files\": []}"}},
        )

    transport = httpx.MockTransport(handler)
    config = OllamaConfig(endpoint_url="http://localhost:11434", model="llama3")

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaClient(config, client=http_client)
        await client.generate(GenerationRequest(idea="build a hello world script"))

    body = captured["body"]
    assert isinstance(body, dict)
    assert "format" in body, "generation request must carry a structured-output format"
    assert body["format"] == GenerationResult.model_json_schema()
