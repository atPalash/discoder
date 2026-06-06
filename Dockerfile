# syntax=docker/dockerfile:1

# Discord-Ollama Coding Agent
#
# The entire system runs inside this single container, which is the
# host-isolation boundary (see design.md "Overview"). Each Job is confined to
# its own per-Job working directory inside this shared container rather than a
# container-per-Job.

FROM python:3.12-slim AS base

# - PYTHONDONTWRITEBYTECODE: avoid .pyc clutter in the image
# - PYTHONUNBUFFERED: stream logs straight to the container's stdout/stderr
# - PIP_NO_CACHE_DIR / PIP_DISABLE_PIP_VERSION_CHECK: smaller, quieter builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies. `git` is required at runtime by the Git_Manager for
# fetch/branch/commit/push against per-target remotes (design.md "Git
# operations"). ca-certificates is needed for HTTPS Git remotes and the
# Ollama endpoint.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching). Copy only the
# packaging metadata and the package itself so the dependency layer is reused
# unless these change.
COPY pyproject.toml README.md ./
COPY discord_ollama_agent ./discord_ollama_agent

# Install the package and its declared dependencies from pyproject.toml.
RUN pip install .

# Run as a non-root user. The container is the isolation boundary, so we still
# drop privileges inside it as defense in depth.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Entry point is the package's main module (discord_ollama_agent/main.py),
# which wires up config loading, registry load, the Ollama connectivity check,
# and the Discord gateway.
ENTRYPOINT ["python", "-m", "discord_ollama_agent.main"]
