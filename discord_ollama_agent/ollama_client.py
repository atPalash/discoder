"""Ollama_Client: HTTP client for the configured Ollama endpoint.

Stateless async HTTP client (built on :class:`httpx.AsyncClient`) that performs
the startup connectivity check and the per-Job generation calls. All traffic is
sent **only** to the configured ``Ollama_Endpoint`` -- the client never contacts
any other destination, so no idea text or generated content can leak to a
third-party service (Req 3.6, Property 29).

Two responsibilities map onto the design's contract:

- **Connectivity check (Req 2.3, 2.4).** :meth:`OllamaClient.check_connectivity`
  issues ``GET {endpoint}/api/tags`` with a fixed 10-second budget and reports
  whether the configured model is present in the returned tag list. It never
  raises for an unreachable endpoint or an absent model; instead it returns a
  :class:`ConnectivityResult` whose ``detail`` records the failure reason so the
  caller can log it and run in degraded mode until a later check succeeds.
- **Generation (Req 3.3, 3.6, 3.10, 3.11).** :meth:`OllamaClient.generate`
  issues ``POST {endpoint}/api/chat`` with the ``format`` field set to the JSON
  schema derived from :class:`~discord_ollama_agent.models.generation.GenerationResult`
  (``model_json_schema()``), constraining the model to emit schema-shaped JSON.
  The request honours the configured per-request generation timeout
  (30-600s, default 120): a timeout raises
  :class:`~discord_ollama_agent.errors.GenerationTimeout`, and any other error
  response raises :class:`~discord_ollama_agent.errors.OllamaError`. Parsing the
  returned text into a ``GenerationResult`` is intentionally left to the
  ``Coding_Agent`` so a malformed completion can be retried as a normal attempt
  (Req 3.7-3.9).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .errors import GenerationTimeout, OllamaError
from .models.config import OllamaConfig
from .models.generation import GenerationRequest, GenerationResult

__all__ = [
    "ConnectivityResult",
    "RawCompletion",
    "OllamaClient",
]

#: Fixed budget for the startup connectivity check, in seconds (Req 2.4).
CONNECTIVITY_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ConnectivityResult:
    """Outcome of a :meth:`OllamaClient.check_connectivity` probe.

    ``available`` is ``True`` only when the endpoint responded within the
    10-second budget *and* the configured model was present in the tag list.
    ``detail`` always carries a human-readable explanation -- a confirmation when
    available, or the failure reason (timeout, transport error, error status, or
    missing model) to be recorded in the operator log (Req 2.4).
    """

    available: bool
    detail: str


@dataclass(frozen=True)
class RawCompletion:
    """The unparsed text completion returned by ``POST /api/chat``.

    ``content`` is the assistant message content exactly as returned by Ollama.
    The ``Coding_Agent`` parses/validates it into a
    :class:`~discord_ollama_agent.models.generation.GenerationResult`, so parse
    failures can be fed back as a retry rather than handled here (Req 3.7-3.9).
    """

    content: str


_SYSTEM_PROMPT = (
    "You are a coding agent. Generate the complete set of project files that "
    "implement the user's request. Respond ONLY with JSON matching the provided "
    "schema: an object with a 'files' array, where each entry has a 'path' "
    "(a relative file path) and 'content' (the full intended contents of that "
    "file). Always emit whole-file content, never diffs or patches."
)


class OllamaClient:
    """Async HTTP client bound to a single configured Ollama endpoint.

    The client is stateless across calls and targets only the configured
    endpoint. An :class:`httpx.AsyncClient` may be injected for testing
    (e.g. with a recording transport for the single-endpoint egress property);
    when none is supplied one is created lazily and owned by this instance.
    """

    def __init__(
        self,
        config: OllamaConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        # Normalize the configured endpoint to a base URL without a trailing
        # slash so request paths join cleanly. This is the ONLY destination the
        # client ever contacts (Req 3.6, Property 29).
        self._base_url = str(config.endpoint_url).rstrip("/")
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the injected client, or lazily create an owned one."""
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        """Close the underlying client if this instance created it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def check_connectivity(self) -> ConnectivityResult:
        """Confirm the configured model is available at the endpoint (Req 2.3, 2.4).

        Issues ``GET {endpoint}/api/tags`` with a fixed 10-second budget and
        checks the configured model name against the returned tags. Never
        raises: an unreachable endpoint, an error status, a timeout, or an
        absent model all yield ``available=False`` with an explanatory
        ``detail`` so the caller can log the reason and reject Jobs with an
        unavailable-model message until a later check confirms the model.
        """
        url = f"{self._base_url}/api/tags"
        try:
            response = await self._get_client().get(
                url, timeout=CONNECTIVITY_TIMEOUT_S
            )
        except httpx.TimeoutException:
            return ConnectivityResult(
                available=False,
                detail=(
                    f"Ollama connectivity check timed out after "
                    f"{CONNECTIVITY_TIMEOUT_S:g}s contacting {url}"
                ),
            )
        except httpx.HTTPError as exc:
            return ConnectivityResult(
                available=False,
                detail=f"Ollama connectivity check failed contacting {url}: {exc}",
            )

        if response.is_error:
            return ConnectivityResult(
                available=False,
                detail=(
                    f"Ollama connectivity check returned HTTP "
                    f"{response.status_code} from {url}"
                ),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return ConnectivityResult(
                available=False,
                detail=f"Ollama connectivity check returned invalid JSON from {url}: {exc}",
            )

        available_tags = self._extract_model_tags(payload)
        if self._model_present(available_tags):
            return ConnectivityResult(
                available=True,
                detail=f"Confirmed model {self._config.model!r} is available at {self._base_url}",
            )
        return ConnectivityResult(
            available=False,
            detail=(
                f"Configured model {self._config.model!r} not found among "
                f"available models {sorted(available_tags)} at {self._base_url}"
            ),
        )

    async def generate(self, request: GenerationRequest) -> RawCompletion:
        """Generate code for ``request`` via ``POST {endpoint}/api/chat`` (Req 3.3, 3.6, 3.10, 3.11).

        The request carries the structured-output ``format`` set to
        ``GenerationResult.model_json_schema()`` so the model is constrained to
        emit schema-shaped JSON, and is sent only to the configured endpoint.
        A read/connect timeout beyond the configured ``generation_timeout_s``
        raises :class:`GenerationTimeout`; any other error response raises
        :class:`OllamaError`.
        """
        url = f"{self._base_url}/api/chat"
        body = {
            "model": self._config.model,
            "messages": self._build_messages(request),
            "format": GenerationResult.model_json_schema(),
            "stream": False,
        }
        try:
            response = await self._get_client().post(
                url, json=body, timeout=float(self._config.generation_timeout_s)
            )
        except httpx.TimeoutException as exc:
            raise GenerationTimeout(
                f"Ollama generation request to {url} exceeded the configured "
                f"timeout of {self._config.generation_timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama generation request to {url} failed: {exc}"
            ) from exc

        if response.is_error:
            raise OllamaError(
                f"Ollama generation request to {url} returned HTTP "
                f"{response.status_code}: {self._error_detail(response)}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaError(
                f"Ollama generation request to {url} returned invalid JSON: {exc}"
            ) from exc

        # Ollama may surface a logical error in a 200 body via an "error" field.
        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            raise OllamaError(f"Ollama endpoint reported an error: {error}")

        return RawCompletion(content=self._extract_content(payload))

    def _build_messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        """Assemble the chat messages from the generation request.

        A fixed system prompt establishes the structured-output contract; the
        idea, any budget-bounded existing-file context, a prior error (for a
        retry), and revision feedback are folded into a single user message.
        """
        parts: list[str] = [f"Idea:\n{request.idea}"]

        if request.context_files:
            rendered = "\n\n".join(
                f"--- {entry.path} ---\n{entry.content}"
                for entry in request.context_files
            )
            parts.append(f"Existing project files:\n{rendered}")

        if request.feedback:
            parts.append(f"Revision feedback:\n{request.feedback}")

        if request.prior_error:
            parts.append(
                "The previous attempt failed with the following error. "
                f"Address it in this attempt:\n{request.prior_error}"
            )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    def _model_present(self, available_tags: set[str]) -> bool:
        """Return whether the configured model matches an available tag.

        Ollama reports tags as ``name`` or ``name:tag`` (e.g. ``llama3:latest``).
        A configured bare name matches its ``:latest`` tag, and a fully
        qualified configured name matches exactly.
        """
        model = self._config.model
        if model in available_tags:
            return True
        if ":" not in model and f"{model}:latest" in available_tags:
            return True
        return False

    @staticmethod
    def _extract_model_tags(payload: object) -> set[str]:
        """Extract the set of model names from an ``/api/tags`` response."""
        tags: set[str] = set()
        if isinstance(payload, dict):
            models = payload.get("models")
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict):
                        name = model.get("name")
                        if isinstance(name, str):
                            tags.add(name)
        return tags

    @staticmethod
    def _extract_content(payload: object) -> str:
        """Pull the assistant message content from an ``/api/chat`` response."""
        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return ""

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Best-effort extraction of an error message from a response body."""
        try:
            payload = response.json()
        except ValueError:
            return response.text
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            return payload["error"]
        return response.text
