"""Entry point and startup wiring for the Discord-Ollama Coding Agent.

Implements the Startup Sequence from the design:

1. Load and validate the system configuration; a :class:`StartupError` is logged
   (its message identifies the offending value) and the process exits non-zero
   without ever reaching a ready state (Req 2.2, 2.5, 4.9, 4.14, 7.6, 13.5).
2. Load the persisted :class:`TargetRegistry`, partitioning it into usable and
   excluded targets; every excluded target is logged with its reason and only
   usable targets are exposed to Jobs (Req 12.1, 12.2).
3. Run the Ollama connectivity check. If the configured model is not confirmed
   within the fixed 10-second budget the System enters *degraded mode*: the
   failure reason is logged and every newly submitted Job is rejected with an
   unavailable-model message until a later connectivity check confirms the model
   (Req 2.3, 2.4).
4. Construct and wire the components -- :class:`ExecutionSandbox` (one per Job),
   :class:`GitManager`, :class:`OllamaClient`, :class:`CodingAgent`,
   :class:`JobManager` (dispatcher loop), and :class:`DiscordBot` (gateway
   connection + slash-command registration + the JobEvent sink wired to
   :meth:`DiscordBot.on_job_event`).

The :mod:`discord` dependency is imported lazily inside :func:`run_gateway`, so
this module imports cleanly even when discord.py is not installed (for example
when only the startup-wiring unit logic is under test).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .coding_agent import CodingAgent
from .config_loader import ConfigLoader
from .discord_bot import DiscordBot, GateContext, MessageSink
from .errors import StartupError
from .execution_sandbox import CancellationToken, ExecutionSandbox
from .git_manager import GitManager
from .job_manager import JobManager
from .models.config import SystemConfig
from .models.job import Job, JobEvent
from .ollama_client import ConnectivityResult, OllamaClient
from .target_registry import RegistryLoadResult, TargetRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only; discord is an optional import.
    import discord

logger = logging.getLogger(__name__)

__all__ = [
    "ConnectivityMonitor",
    "Application",
    "UNAVAILABLE_MODEL_MESSAGE",
    "EXIT_STARTUP_ERROR",
    "build_application",
    "run_gateway",
    "async_main",
    "main",
]

# -- Environment / defaults ---------------------------------------------------

#: Environment variable naming the system configuration YAML file.
ENV_CONFIG_PATH = "DISCORD_OLLAMA_CONFIG"
#: Environment variable naming the persisted Target_Registry JSON file.
ENV_REGISTRY_PATH = "DISCORD_OLLAMA_REGISTRY"
#: Environment variable carrying the Discord bot token used to connect the gateway.
ENV_DISCORD_TOKEN = "DISCORD_BOT_TOKEN"
#: Environment variable overriding the degraded-mode connectivity recheck interval.
ENV_RECHECK_INTERVAL = "DISCORD_OLLAMA_CONNECTIVITY_RECHECK_S"

#: Default configuration path when the environment variable is unset.
DEFAULT_CONFIG_PATH = "config.yaml"
#: Default registry filename, resolved relative to the workspace directory.
DEFAULT_REGISTRY_FILENAME = "targets.json"
#: Default interval, in seconds, between degraded-mode connectivity rechecks.
DEFAULT_RECHECK_INTERVAL_S = 30.0
#: Subdirectory of the workspace under which per-Job working copies are created.
WORKING_SUBDIR = ".agent-jobs"

#: Process exit code used when fail-fast startup validation rejects the config
#: or registry (Req 2.2, 2.5, 4.9, 4.14, 7.6, 13.5).
EXIT_STARTUP_ERROR = 1

#: Reply sent for a Job submitted while the System is in degraded mode because
#: the configured Ollama model could not be confirmed available (Req 2.4).
UNAVAILABLE_MODEL_MESSAGE = (
    "The configured coding model is currently unavailable; the system is running "
    "in degraded mode and cannot accept new jobs until connectivity is restored."
)


class ConnectivityMonitor:
    """Tracks Ollama model availability and recovers from degraded mode.

    The initial :meth:`check_once` establishes whether the configured model is
    available. When it is not, the System runs in *degraded mode* and rejects new
    Jobs (Req 2.4); :meth:`start` then launches a background loop that re-runs the
    connectivity check on a fixed interval until the model is confirmed, at which
    point degraded mode is cleared and the loop stops (Req 2.4).
    """

    def __init__(
        self,
        client: OllamaClient,
        recheck_interval_s: float = DEFAULT_RECHECK_INTERVAL_S,
    ) -> None:
        self._client = client
        self._recheck_interval_s = recheck_interval_s
        self._available = False
        self._detail = "connectivity has not yet been checked"
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    @property
    def available(self) -> bool:
        """Whether the configured model is currently confirmed available.

        ``False`` means the System is in degraded mode and new Jobs must be
        rejected with the unavailable-model message (Req 2.4).
        """
        return self._available

    @property
    def detail(self) -> str:
        """The most recent connectivity detail (confirmation or failure reason)."""
        return self._detail

    def _apply(self, result: ConnectivityResult) -> None:
        self._available = result.available
        self._detail = result.detail

    async def check_once(self) -> ConnectivityResult:
        """Run a single connectivity check and update the availability flag.

        Logs a confirmation when the model is available, or the failure reason
        when it is not, so the operator log records why the System entered (or
        remains in) degraded mode (Req 2.4).
        """
        result = await self._client.check_connectivity()
        self._apply(result)
        if result.available:
            logger.info("Ollama connectivity confirmed: %s", result.detail)
        else:
            logger.error(
                "Ollama connectivity check failed; entering degraded mode: %s",
                result.detail,
            )
        return result

    def start(self) -> None:
        """Start the background recheck loop if the System is degraded.

        When the model is already available there is nothing to recover, so no
        loop is started. Must be called from within the running event loop.
        """
        if self._available:
            return
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._recheck_loop())

    async def stop(self) -> None:
        """Stop the background recheck loop and wait for it to unwind."""
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _recheck_loop(self) -> None:
        """Re-run the connectivity check until the model is confirmed (Req 2.4)."""
        while not self._stopping and not self._available:
            try:
                await asyncio.sleep(self._recheck_interval_s)
            except asyncio.CancelledError:
                raise
            if self._stopping:
                break
            await self.check_once()
        if self._available:
            logger.info("Ollama model confirmed available; leaving degraded mode")


class Application:
    """Wires the orchestration and execution layers behind the Discord gateway.

    Owns the long-lived collaborators -- the :class:`GitManager`,
    :class:`OllamaClient`, the :class:`ConnectivityMonitor`, and the
    :class:`JobManager` whose dispatcher loop drives Jobs. Each Job is executed by
    a freshly constructed :class:`ExecutionSandbox` + :class:`CodingAgent` so the
    per-Job working-copy roots stay isolated even when several Jobs run
    concurrently (Property 5). The :class:`DiscordBot` and its lifecycle event
    sink are attached once the gateway client exists, breaking the
    bot <-> Job_Manager construction cycle.
    """

    def __init__(
        self,
        config: SystemConfig,
        registry: TargetRegistry,
        registry_result: RegistryLoadResult,
        ollama_client: OllamaClient,
        *,
        recheck_interval_s: float = DEFAULT_RECHECK_INTERVAL_S,
    ) -> None:
        self.config = config
        self.registry = registry
        self.registry_result = registry_result
        self.ollama_client = ollama_client
        self.monitor = ConnectivityMonitor(ollama_client, recheck_interval_s)

        # Stateless per-target Git operations shared across Jobs.
        self._git_manager = GitManager(push_timeout_s=config.push_timeout_s)
        # Parent directory under which each Job's isolated working copy is created.
        self._working_root = Path(config.workspace_dir) / WORKING_SUBDIR

        # Indirection so the JobManager/CodingAgent event sink can be pointed at
        # the DiscordBot after the bot is constructed (the bot needs the
        # JobManager, and the JobManager needs the bot's sink).
        self._event_sink: Optional[JobEventSinkFn] = None

        self.job_manager = JobManager(
            concurrency_limit=config.concurrency_limit,
            worker=self._worker,
            event_sink=self._dispatch_event,
        )
        self.bot: DiscordBot | None = None

    async def _dispatch_event(self, event: JobEvent) -> None:
        """Forward a :class:`JobEvent` to the attached bot sink, if any."""
        sink = self._event_sink
        if sink is not None:
            await sink(event)

    async def _worker(self, job: Job, cancel: CancellationToken) -> None:
        """Execute one Job with its own sandbox and agent (Req 7.2, Property 5).

        A fresh :class:`ExecutionSandbox` (rooted under the workspace) and a
        :class:`CodingAgent` are constructed per Job so concurrently Running Jobs
        never share working-copy state, then :meth:`CodingAgent.run` drives the
        full lifecycle.
        """
        sandbox = ExecutionSandbox(
            self._working_root,
            build_output_cap_bytes=self.config.build_output_cap_bytes,
        )
        agent = CodingAgent(
            sandbox=sandbox,
            ollama_client=self.ollama_client,
            git_manager=self._git_manager,
            config=self.config,
            registry=self.registry,
            event_sink=self._dispatch_event,
        )
        await agent.run(job, cancel)

    def attach_bot(self, bot: DiscordBot) -> None:
        """Wire the Job lifecycle event sink to ``bot.on_job_event`` (Req 6.1-6.3)."""
        self.bot = bot
        self._event_sink = bot.on_job_event

    async def check_connectivity(self) -> ConnectivityResult:
        """Run the initial Ollama connectivity check (Req 2.3, 2.4)."""
        return await self.monitor.check_once()

    def start_dispatcher(self) -> None:
        """Start the Job_Manager dispatcher loop (Req 7.3-7.5)."""
        self.job_manager.start()

    async def shutdown(self) -> None:
        """Tear down the dispatcher, recheck loop, and HTTP client."""
        await self.monitor.stop()
        await self.job_manager.stop()
        await self.ollama_client.aclose()


# A Job lifecycle event sink: an async callback receiving a :class:`JobEvent`.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    JobEventSinkFn = Callable[[JobEvent], Awaitable[None]]


# -- Construction helpers -----------------------------------------------------


def _resolve_config_path(config_path: str | os.PathLike[str] | None) -> Path:
    """Resolve the system configuration path from the argument or environment."""
    if config_path is not None:
        return Path(config_path)
    return Path(os.environ.get(ENV_CONFIG_PATH, DEFAULT_CONFIG_PATH))


def _resolve_registry_path(
    registry_path: str | os.PathLike[str] | None, config: SystemConfig
) -> Path:
    """Resolve the registry path from the argument, environment, or workspace."""
    if registry_path is not None:
        return Path(registry_path)
    env_value = os.environ.get(ENV_REGISTRY_PATH)
    if env_value:
        return Path(env_value)
    return Path(config.workspace_dir) / DEFAULT_REGISTRY_FILENAME


def _recheck_interval() -> float:
    """Read the degraded-mode recheck interval from the environment."""
    raw = os.environ.get(ENV_RECHECK_INTERVAL)
    if not raw:
        return DEFAULT_RECHECK_INTERVAL_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using default %.0fs",
            ENV_RECHECK_INTERVAL,
            raw,
            DEFAULT_RECHECK_INTERVAL_S,
        )
        return DEFAULT_RECHECK_INTERVAL_S
    return value if value > 0 else DEFAULT_RECHECK_INTERVAL_S


def build_application(
    config_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] | None = None,
    *,
    ollama_client: OllamaClient | None = None,
    recheck_interval_s: float | None = None,
) -> Application:
    """Run the fail-fast startup sequence and build the wired :class:`Application`.

    Loads and validates the system config (raising :class:`StartupError` on any
    invalid value, with a message identifying it; Req 2.2, 2.5, 4.9, 4.14, 7.6,
    13.5) and loads the persisted registry, logging every excluded target with
    its reason and exposing only usable targets (Req 12.1, 12.2). The returned
    application has its collaborators wired but has not yet run the connectivity
    check, started the dispatcher, or connected the gateway.

    Raises:
        StartupError: If the configuration or registry file is invalid.
    """
    resolved_config_path = _resolve_config_path(config_path)
    logger.info("Loading system configuration from %s", resolved_config_path)
    config = ConfigLoader().load(resolved_config_path)

    resolved_registry_path = _resolve_registry_path(registry_path, config)
    logger.info("Loading target registry from %s", resolved_registry_path)
    registry = TargetRegistry()
    load_result = registry.load(resolved_registry_path, Path(config.workspace_dir))

    usable = registry.names()
    logger.info(
        "Target registry loaded: %d usable, %d excluded",
        len(load_result.usable),
        len(load_result.excluded),
    )
    if usable:
        logger.info("Usable targets: %s", ", ".join(usable))
    # Each excluded target was already logged with its reason by the registry;
    # surface a concise summary as well.
    for excluded in load_result.excluded:
        logger.warning(
            "Excluded target %r: %s", excluded.name, excluded.reason
        )

    client = ollama_client or OllamaClient(config.ollama)
    interval = (
        recheck_interval_s if recheck_interval_s is not None else _recheck_interval()
    )
    return Application(
        config=config,
        registry=registry,
        registry_result=load_result,
        ollama_client=client,
        recheck_interval_s=interval,
    )


# -- Discord gateway adapters -------------------------------------------------


class _InteractionContext:
    """Adapts a discord.py ``Interaction`` to the bot's :class:`GateContext`.

    The :class:`DiscordBot` command handlers only need the originating channel id
    and the invoking user id; this exposes exactly those from a live interaction
    so the handlers stay decoupled from the concrete library type.
    """

    def __init__(self, interaction: "discord.Interaction") -> None:
        self._interaction = interaction

    @property
    def channel_id(self) -> int:
        return int(self._interaction.channel_id or 0)

    @property
    def user_id(self) -> int:
        return int(self._interaction.user.id)


class _DiscordMessageSink:
    """A :class:`MessageSink` that posts lifecycle messages to a channel.

    Resolves the target channel from the connected client (falling back to a
    fetch when it is not cached) and sends the rendered status text. A failed
    send raises, so the bot's bounded delivery-retry logic can react (Req 6.7).
    """

    def __init__(self, client: "discord.Client") -> None:
        self._client = client

    async def send(self, channel_id: int, content: str) -> None:
        channel = self._client.get_channel(channel_id)
        if channel is None:
            channel = await self._client.fetch_channel(channel_id)
        await channel.send(content)  # type: ignore[union-attr]


def _register_commands(
    tree: "discord.app_commands.CommandTree",
    bot: DiscordBot,
    monitor: ConnectivityMonitor,
) -> None:
    """Register the six slash commands on ``tree``, delegating to the bot.

    Every handler builds a :class:`GateContext` from the interaction and defers
    gating/validation/state changes to the corresponding :class:`DiscordBot`
    method, then sends the returned reply within the interaction window. ``/build``
    and ``/revise`` additionally short-circuit with the unavailable-model message
    while the System is in degraded mode (Req 2.4).
    """
    import discord
    from discord import app_commands

    async def _reply(interaction: "discord.Interaction", content: str) -> None:
        await interaction.response.send_message(content, ephemeral=True)

    @tree.command(name="build", description="Generate code for an idea against a target")
    @app_commands.describe(
        idea="The coding idea to implement",
        target="The target to build against (optional if a default is configured)",
    )
    async def build(
        interaction: "discord.Interaction",
        idea: str,
        target: Optional[str] = None,
    ) -> None:
        if not monitor.available:
            await _reply(interaction, UNAVAILABLE_MODEL_MESSAGE)
            return
        ctx = _InteractionContext(interaction)
        await _reply(interaction, await bot.on_build(ctx, target, idea))

    @tree.command(name="status", description="Report the status of a job")
    @app_commands.describe(job_id="The job identifier to query")
    async def status(interaction: "discord.Interaction", job_id: str) -> None:
        ctx = _InteractionContext(interaction)
        await _reply(interaction, await bot.on_status(ctx, job_id))

    @tree.command(name="cancel", description="Cancel a queued or running job")
    @app_commands.describe(job_id="The job identifier to cancel")
    async def cancel(interaction: "discord.Interaction", job_id: str) -> None:
        ctx = _InteractionContext(interaction)
        await _reply(interaction, await bot.on_cancel(ctx, job_id))

    @tree.command(name="targets", description="List the registered targets")
    async def targets(interaction: "discord.Interaction") -> None:
        ctx = _InteractionContext(interaction)
        await _reply(interaction, await bot.on_targets(ctx))

    @tree.command(name="addtarget", description="Register a new target (admin only)")
    @app_commands.describe(
        name="The unique target name",
        path="The target directory path inside the workspace",
        repo_remote="The target's Git remote URL",
        credentials_ref="Reference to the target's Git credentials (env var or path)",
    )
    async def addtarget(
        interaction: "discord.Interaction",
        name: str,
        path: str,
        repo_remote: str = "",
        credentials_ref: str = "",
    ) -> None:
        ctx = _InteractionContext(interaction)
        await _reply(
            interaction,
            await bot.on_addtarget(ctx, name, path, repo_remote, credentials_ref),
        )

    @tree.command(name="revise", description="Revise a completed job on the same branch")
    @app_commands.describe(
        job_id="The succeeded job identifier to revise",
        feedback="The revision feedback to apply",
    )
    async def revise(
        interaction: "discord.Interaction", job_id: str, feedback: str
    ) -> None:
        if not monitor.available:
            await _reply(interaction, UNAVAILABLE_MODEL_MESSAGE)
            return
        ctx = _InteractionContext(interaction)
        await _reply(interaction, await bot.on_revise(ctx, job_id, feedback))


async def run_gateway(app: Application, token: str) -> None:
    """Run the connectivity check, start dispatch, and connect the gateway.

    Runs the initial Ollama connectivity check and (when degraded) starts the
    recovery recheck loop (Req 2.3, 2.4), constructs the gateway client and the
    message sink, attaches the :class:`DiscordBot` and its JobEvent sink, and on
    gateway-ready starts the dispatcher and registers/syncs the slash commands
    before serving interactions. The ``discord`` library is imported here so the
    module imports cleanly without it installed.
    """
    import discord
    from discord import app_commands

    # Phase 3: connectivity check + degraded-mode recovery loop (Req 2.3, 2.4).
    await app.check_connectivity()
    app.monitor.start()

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    # Phase 4: wire the bot (with its message sink) and the JobEvent sink.
    message_sink: MessageSink = _DiscordMessageSink(client)
    bot = DiscordBot(
        config=app.config,
        job_manager=app.job_manager,
        target_registry=app.registry,
        message_sink=message_sink,
    )
    app.attach_bot(bot)

    @client.event
    async def on_ready() -> None:  # pragma: no cover - requires a live gateway
        logger.info("Discord gateway connected as %s", client.user)
        # Start the dispatcher and register commands once the loop is live.
        app.start_dispatcher()
        _register_commands(tree, bot, app.monitor)
        await tree.sync()
        logger.info("Slash commands registered; system is ready")

    try:
        await client.start(token)
    finally:
        await app.shutdown()
        if not client.is_closed():
            await client.close()


async def async_main(
    config_path: str | os.PathLike[str] | None = None,
    registry_path: str | os.PathLike[str] | None = None,
) -> int:
    """Async startup: build the application and connect the gateway.

    Returns a process exit code: :data:`EXIT_STARTUP_ERROR` when fail-fast
    validation rejects the config/registry or the Discord token is absent, and
    ``0`` on a clean shutdown.
    """
    try:
        app = build_application(config_path, registry_path)
    except StartupError as exc:
        # The message identifies the offending value; log it and refuse to start
        # (no ready state) (Req 2.2, 2.5, 4.9, 4.14, 7.6, 13.5).
        logger.error("Startup aborted due to invalid configuration: %s", exc)
        return EXIT_STARTUP_ERROR

    token = os.environ.get(ENV_DISCORD_TOKEN)
    if not token:
        logger.error(
            "Startup aborted: the Discord bot token is missing (set %s)",
            ENV_DISCORD_TOKEN,
        )
        await app.shutdown()
        return EXIT_STARTUP_ERROR

    await run_gateway(app, token)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Synchronous entry point: configure logging and run the startup sequence.

    Accepts an optional configuration path as the first CLI argument (otherwise
    the ``DISCORD_OLLAMA_CONFIG`` environment variable or the default is used).
    Returns the process exit code so ``python -m discord_ollama_agent.main`` exits
    non-zero on a startup failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = sys.argv[1:] if argv is None else argv
    config_path = args[0] if args else None
    return asyncio.run(async_main(config_path))


if __name__ == "__main__":  # pragma: no cover - exercised via the container entry point
    sys.exit(main())
