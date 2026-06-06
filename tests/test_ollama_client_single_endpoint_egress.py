"""Property-based test for single-endpoint egress (task 6.2).

Exercises :meth:`OllamaClient.generate` against a recording ``httpx``
transport and asserts the single-endpoint egress invariant: every network
destination the client contacts while servicing a generation request is the
configured ``Ollama_Endpoint`` and nothing else. No idea text or generated
content can therefore leak to a third-party destination (Req 3.6, Property 29).

The recording transport answers every request with a well-formed completion so
the request actually completes through the production code path; only the
network is replaced. Each request's URL (scheme/host/port and path origin) is
captured and checked against the configured endpoint across a wide space of
endpoints and request payloads (ideas, context files, feedback, prior errors).
"""

from __future__ import annotations

import httpx
from hypothesis import given
from hypothesis import strategies as st

from discord_ollama_agent.models.config import OllamaConfig
from discord_ollama_agent.models.generation import FileEntry, GenerationRequest
from discord_ollama_agent.ollama_client import OllamaClient

# Exclude lone surrogate code points (Unicode category "Cs"): they are not
# valid in well-formed text and are outside the space of real ideas, file
# paths, and contents, so constraining them keeps the generator within the
# valid input space rather than testing serializer-level edge cases.
_text = st.text(alphabet=st.characters(blacklist_categories=["Cs"]), max_size=200)

_file_entries = st.builds(FileEntry, path=_text, content=_text)

_generation_requests = st.builds(
    GenerationRequest,
    idea=_text,
    context_files=st.lists(_file_entries, max_size=5),
    prior_error=st.none() | _text,
    feedback=st.none() | _text,
)

# A range of syntactically valid endpoints: differing schemes, hosts, and
# explicit/implicit ports, so the property holds wherever the operator points
# the client.
_endpoints = st.sampled_from(
    [
        "http://localhost:11434",
        "https://ollama.internal:443",
        "http://127.0.0.1:8080",
        "https://gpu-box.example.com",
        "http://10.0.0.5:11434/ollama",
        "https://ollama.local:11434/",
    ]
)


# Feature: discord-ollama-coding-agent, Property 29
# Property 29: Generation traffic targets only the configured endpoint.
# Validates: Requirements 3.6
@given(endpoint=_endpoints, request=_generation_requests)
async def test_generation_traffic_targets_only_configured_endpoint(
    endpoint: str, request: GenerationRequest
) -> None:
    """For any generation request, the only destination contacted is the
    configured Ollama_Endpoint; no traffic goes anywhere else (Req 3.6)."""
    config = OllamaConfig(endpoint_url=endpoint, model="llama3")
    expected = httpx.URL(endpoint)

    contacted: list[httpx.URL] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        # Record the destination of every outbound request before answering it.
        contacted.append(http_request.url)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"files": []}'}},
        )

    transport = httpx.MockTransport(handler)
    client = OllamaClient(config, client=httpx.AsyncClient(transport=transport))
    try:
        await client.generate(request)
    finally:
        await client.aclose()

    # At least one request must have been made (the generation call itself).
    assert contacted, "generate() contacted no endpoint"

    expected_origin = (
        expected.scheme,
        expected.host,
        expected.port if expected.port is not None else None,
    )
    for url in contacted:
        actual_origin = (
            url.scheme,
            url.host,
            url.port if url.port is not None else None,
        )
        assert actual_origin == expected_origin, (
            f"generation traffic contacted {url} but the only permitted "
            f"destination is the configured endpoint {expected}"
        )
        # The path must stay under the configured endpoint's base path, so no
        # request is redirected to an unrelated origin via path manipulation.
        assert url.path.startswith(expected.path.rstrip("/")), (
            f"contacted path {url.path!r} is outside the configured endpoint "
            f"base path {expected.path!r}"
        )
