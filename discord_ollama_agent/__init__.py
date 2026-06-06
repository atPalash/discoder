"""Discord-Ollama Coding Agent.

A single-container Python application that turns coding ideas submitted through
Discord slash commands into committed, pushed Git branches using a locally hosted
Ollama model.

This package is organized into:

- ``models``            -- pydantic data models (config, jobs, targets, generation).
- ``config_loader``     -- fail-fast system configuration loading/validation.
- ``target_registry``   -- persisted set of Registered_Targets.
- ``ollama_client``     -- HTTP client for the configured Ollama endpoint.
- ``execution_sandbox`` -- per-Job working-directory boundary + subprocess runner.
- ``git_manager``       -- per-target sync/branch/commit/push operations.
- ``coding_agent``      -- the iterative generate -> write -> build -> fix loop.
- ``job_manager``       -- Job identity, queue, concurrency, and lifecycle.
- ``discord_bot``       -- Discord transport, gating, commands, and status sink.
- ``main``              -- startup wiring and entry point.
- ``errors``            -- shared exception hierarchy.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
