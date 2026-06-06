"""Discord_Bot: the Discord transport layer.

Owns the gateway connection, registers the six slash commands, performs channel
and authorization gating, validates command arguments, and renders all outbound
status/reply messages.

This module currently implements the centralized command gate:

- :class:`GateDecision`  -- the enumerated outcome of a gate evaluation
  (allowed, channel-not-permitted, or user-not-authorized).
- :class:`GateResult`    -- the immutable result of :meth:`DiscordBot._gate`,
  carrying whether the command may proceed and, on denial, a ready-to-send
  reply message.
- :meth:`DiscordBot._gate` -- enforces the gating conjunction that every command
  must satisfy before it is acted upon: the originating channel is among the
  configured ``Allowed_Channels`` **and** the invoking user is authorized for
  that command (the admin allowlist for ``/addtarget``, the user allowlist for
  every other command). Both checks must pass; otherwise the command produces a
  denial reply and makes no state change (Req 1.4, 6.6, 8.1, 9.3, 10.2, 11.5,
  13.1, 13.2, 13.3, Property 1).

The slash command handlers, argument validation, and the Job lifecycle event
sink are added in a later task. The gate accepts any object exposing
``channel_id`` and ``user_id`` (see :class:`GateContext`) so it is testable
without a live Discord gateway connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .errors import (
    DuplicateTargetError,
    PathOutsideWorkspaceError,
    TargetDirectoryNotFoundError,
)
from .job_manager import CancelResult, JobManager
from .models.config import SystemConfig
from .models.job import JobEvent, JobStatus
from .models.target import TARGET_NAME_MAX_LENGTH, RegisteredTarget
from .target_registry import TargetRegistry

__all__ = [
    "GateContext",
    "GateDecision",
    "GateResult",
    "MessageSink",
    "DiscordBot",
]

logger = logging.getLogger(__name__)

#: Maximum number of delivery retries attempted after an initial failed status
#: post before the Discord_Bot gives up and logs a delivery-failure reason
#: (Req 6.7, Property 22). The initial attempt plus at most this many retries
#: bounds the total attempts.
_MAX_DELIVERY_RETRIES = 3


class GateContext(Protocol):
    """The minimal slice of a Discord interaction the gate needs.

    Discord's ``Interaction`` object carries far more than the gate uses; the
    gate only needs the originating channel identifier and the invoking user
    identifier. Depending on this narrow Protocol (rather than the concrete
    library type) keeps :meth:`DiscordBot._gate` pure and unit-testable without a
    live gateway connection -- any object exposing these two integer attributes
    can be gated.
    """

    @property
    def channel_id(self) -> int: ...

    @property
    def user_id(self) -> int: ...


class MessageSink(Protocol):
    """An awaitable channel message poster the Discord_Bot delivers through.

    Lifecycle status messages (started/succeeded/failed) are posted to the
    originating channel through this narrow async interface rather than the
    concrete Discord client, so :meth:`DiscordBot.on_job_event` and its bounded
    delivery-retry logic can be exercised with a fake sink that fails a chosen
    number of times (Req 6.1-6.3, 6.7, Property 22). An implementation raises on
    a failed delivery and returns normally on success.
    """

    async def send(self, channel_id: int, content: str) -> None: ...


class GateDecision(Enum):
    """The mutually exclusive outcomes of a gate evaluation.

    The channel restriction is evaluated before the user authorization check, so
    a command invoked from a disallowed channel reports
    :attr:`CHANNEL_NOT_PERMITTED` regardless of the user's allowlist membership
    (Req 13.1, 13.3).
    """

    #: Both checks passed; the command may be acted upon.
    ALLOWED = "allowed"
    #: The originating channel is not among the configured Allowed_Channels.
    CHANNEL_NOT_PERMITTED = "channel_not_permitted"
    #: The channel is allowed but the user is not authorized for the command.
    USER_NOT_AUTHORIZED = "user_not_authorized"


@dataclass(frozen=True)
class GateResult:
    """The immutable result of evaluating the command gate.

    Fields:
        decision: Which of the three :class:`GateDecision` outcomes applied.
        message: ``None`` when :attr:`allowed` is true; otherwise the
            ready-to-send denial reply explaining why the command was refused
            (Req 13.1 for the channel case, Req 1.4/6.6/8.1/9.3/10.2/11.5 for
            the authorization case).
    """

    decision: GateDecision
    message: str | None = None

    @property
    def allowed(self) -> bool:
        """True iff the command passed both the channel and user checks."""

        return self.decision is GateDecision.ALLOWED


#: Reply sent when a command is invoked outside the Allowed_Channels (Req 13.1).
_CHANNEL_DENIED_MESSAGE = "This command is not permitted in this channel."

#: Reply sent when an ordinary command's invoker is not on the user allowlist
#: (Req 1.4, 6.6, 8.1, 9.3, 11.5).
_USER_DENIED_MESSAGE = "You are not authorized to run this command."

#: Reply sent when ``/addtarget`` is invoked by a user not on the admin
#: allowlist (Req 10.2).
_ADMIN_DENIED_MESSAGE = "You are not authorized to register targets."


#: Reply requesting a non-empty idea when the ``/build`` idea is whitespace-only
#: (Req 1.5, Property 2).
_EMPTY_IDEA_MESSAGE = "Please provide a non-empty idea."

#: Reply requesting non-empty feedback when the ``/revise`` feedback is
#: whitespace-only (Req 11.8, Property 2).
_EMPTY_FEEDBACK_MESSAGE = "Please provide non-empty feedback."

#: Reply when ``/build`` omits a target and no Default_Target is configured
#: (Req 1.7).
_NO_TARGET_MESSAGE = (
    "No target specified and no default target is configured; "
    "please specify a target."
)

#: Reply when an ``/addtarget`` name is blank or longer than the allowed bound
#: (Req 10.6).
_INVALID_TARGET_NAME_MESSAGE = (
    "The target name is invalid; it must contain at least one non-whitespace "
    f"character and be at most {TARGET_NAME_MAX_LENGTH} characters."
)


def _not_found_message(job_id: str) -> str:
    """Render a not-found reply that echoes the queried Job id (Property 21).

    Shared by ``/status``, ``/cancel``, and ``/revise`` so a reference to a Job
    that does not exist always replies with a message identifying the requested
    identifier and makes no state change (Req 6.5, 8.2, 11.6).
    """
    return f"No job found with id `{job_id}`."


def _render_event(event: JobEvent) -> str | None:
    """Render the Discord status message for a Job lifecycle ``event``.

    Returns the message text for the started (Running), completion (Succeeded),
    and failure (Failed) transitions (Req 6.1, 6.2, 6.3), or ``None`` for a
    transition that has no Discord message (Queued, Cancelled). The Succeeded
    message includes either the pushed branch name or the no-changes note
    (Req 6.2); the Failed message includes the recorded failure reason (Req 6.3).
    Every message references the Job identifier.
    """
    if event.status is JobStatus.RUNNING:
        return f"Job `{event.job_id}` has started."
    if event.status is JobStatus.SUCCEEDED:
        if event.branch_name:
            detail = f"pushed branch `{event.branch_name}`"
        else:
            detail = event.result_note or "no changes were produced"
        return f"Job `{event.job_id}` succeeded: {detail}."
    if event.status is JobStatus.FAILED:
        reason = event.failure_reason or "unknown failure"
        return f"Job `{event.job_id}` failed: {reason}."
    # Queued/Cancelled transitions have no Discord status message.
    return None


class DiscordBot:
    """Discord transport layer: gateway, slash commands, gating, and rendering.

    This class currently implements the centralized command gate. Slash command
    registration, argument validation, message rendering, and the Job lifecycle
    event sink are added in a later task.
    """

    def __init__(
        self,
        config: SystemConfig,
        job_manager: JobManager | None = None,
        target_registry: TargetRegistry | None = None,
        message_sink: MessageSink | None = None,
    ) -> None:
        # The validated system configuration supplies the gating allowlists and
        # the set of Allowed_Channels. At least one Allowed_Channel is
        # guaranteed by SystemConfig construction (Req 13.5).
        self._config = config
        # The orchestration layer the command handlers drive: minting/looking-up
        # Jobs and revisions and cancelling them (Req 1.1, 6.4, 8.x, 11.x).
        self._job_manager = job_manager
        # The usable-target store used to resolve /build targets, list targets,
        # and register new ones (Req 1.6, 9.1, 10.1).
        self._target_registry = target_registry
        # The async channel poster lifecycle status messages are delivered
        # through; abstracted so delivery (and its bounded retry) is testable
        # without a live gateway (Req 6.1-6.3, 6.7).
        self._message_sink = message_sink

    def _gate(self, itx: GateContext, *, admin: bool) -> GateResult:
        """Evaluate the gating conjunction for a single command invocation.

        A command is acted upon **if and only if** the originating channel is
        among the configured ``Allowed_Channels`` *and* the invoking user is
        authorized for that command: membership in the admin allowlist when
        ``admin`` is true (``/addtarget``), otherwise membership in the user
        allowlist (every other command). Both conditions must hold; if either
        fails the returned :class:`GateResult` is not allowed and carries the
        appropriate denial reply, and the caller must make no state change
        (Req 1.4, 6.6, 8.1, 9.3, 10.2, 11.5, 13.1, 13.2, 13.3, Property 1).

        The channel restriction is checked first, so an unauthorized user
        invoking a command from a disallowed channel receives the
        not-permitted-here reply (Req 13.1, 13.3).

        Args:
            itx: The interaction context exposing ``channel_id`` and
                ``user_id`` (see :class:`GateContext`).
            admin: ``True`` for ``/addtarget``, which is gated on the admin
                allowlist; ``False`` for every other command, gated on the user
                allowlist.

        Returns:
            A :class:`GateResult` whose :attr:`~GateResult.allowed` is true only
            when both checks pass; otherwise a denial result with a populated
            ``message``.
        """

        # 1) Channel check first: a command from a disallowed channel is refused
        #    regardless of the invoker's authorization (Req 13.1, 13.3).
        if itx.channel_id not in self._config.allowed_channels:
            return GateResult(
                decision=GateDecision.CHANNEL_NOT_PERMITTED,
                message=_CHANNEL_DENIED_MESSAGE,
            )

        # 2) User authorization against the allowlist appropriate to the command
        #    (admin allowlist for /addtarget, user allowlist otherwise)
        #    (Req 1.4, 6.6, 8.1, 9.3, 10.2, 11.5).
        allowlist = self._config.admin_users if admin else self._config.authorized_users
        if itx.user_id not in allowlist:
            return GateResult(
                decision=GateDecision.USER_NOT_AUTHORIZED,
                message=_ADMIN_DENIED_MESSAGE if admin else _USER_DENIED_MESSAGE,
            )

        # 3) Both the channel and the user checks passed (Req 13.2, 13.3).
        return GateResult(decision=GateDecision.ALLOWED)

    # -- Command handlers ------------------------------------------------------
    #
    # Each handler first runs the gate (Property 1); on denial it returns the
    # gate's ready-to-send reply and makes no state change. On success it
    # validates its arguments, drives the Job_Manager / Target_Registry, and
    # returns the rendered reply string. Returning the reply (rather than posting
    # it through a live gateway) keeps the handlers pure and unit-testable; the
    # caller wiring the real gateway sends the returned string as the interaction
    # response, always within the 5-second interaction window.

    async def on_build(
        self, itx: GateContext, target: str | None, idea: str
    ) -> str:
        """Handle ``/build``: validate inputs, resolve the target, queue a Job.

        Gated on the user allowlist (Req 1.4). The idea must contain a
        non-whitespace character (Req 1.5, Property 2) and, where a maximum idea
        length is configured, must not exceed it in Unicode characters
        (Req 1.8, Property 3). The target is resolved from the supplied name or,
        when omitted, from the configured Default_Target (Req 1.2, 1.7); an
        unresolvable name is rejected as not-registered (Req 1.6, Property 23).
        On success a Queued Job is created and the reply carries its identifier,
        which contains the target name (Req 1.1, 1.3).
        """
        gate = self._gate(itx, admin=False)
        if not gate.allowed:
            return gate.message or _USER_DENIED_MESSAGE

        if not idea.strip():
            return _EMPTY_IDEA_MESSAGE

        max_len = self._config.max_idea_length
        if max_len is not None and len(idea) > max_len:
            return (
                f"The idea exceeds the configured maximum length of {max_len} "
                "characters."
            )

        # Resolve the target: an omitted/blank name routes to the Default_Target
        # (Req 1.2, 1.7).
        name = target.strip() if target is not None else ""
        if not name:
            if not self._config.default_target:
                return _NO_TARGET_MESSAGE
            name = self._config.default_target

        registry = self._require_registry()
        resolved = registry.get(name)
        if resolved is None:
            return f"Target {name!r} is not registered."

        manager = self._require_manager()
        job = manager.create_job(
            resolved, idea, channel_id=itx.channel_id, user_id=itx.user_id
        )
        manager.notify()
        return f"Created job `{job.id}` (Queued) for target {resolved.name!r}."

    async def on_status(self, itx: GateContext, job_id: str) -> str:
        """Handle ``/status``: report a Job's current status (Req 6.4).

        Gated on the user allowlist; an unauthorized invoker is refused without
        disclosing any status (Req 6.6). An unknown id yields a not-found reply
        that echoes the queried identifier (Req 6.5, Property 21).
        """
        gate = self._gate(itx, admin=False)
        if not gate.allowed:
            return gate.message or _USER_DENIED_MESSAGE

        job = self._require_manager().get(job_id)
        if job is None:
            return _not_found_message(job_id)
        return f"Job `{job.id}` status: {job.status.value}."

    async def on_cancel(self, itx: GateContext, job_id: str) -> str:
        """Handle ``/cancel``: cancel a Job, with a status-dependent reply.

        Gated on the user allowlist (Req 8.1). The outcome depends only on the
        Job's status (delegated to :meth:`JobManager.cancel`): an unknown id is
        reported not-found echoing the identifier (Req 8.2, Property 21), a
        Queued or Running Job is cancelled with a confirmation referencing the id
        (Req 8.3, 8.4), and a terminal Job is rejected as un-cancellable
        (Req 8.6).
        """
        gate = self._gate(itx, admin=False)
        if not gate.allowed:
            return gate.message or _USER_DENIED_MESSAGE

        outcome = await self._require_manager().cancel(job_id)
        if outcome.result is CancelResult.NOT_FOUND:
            return _not_found_message(job_id)
        if outcome.result is CancelResult.REJECTED_TERMINAL:
            return (
                f"Job `{job_id}` cannot be cancelled because it is in a terminal "
                "state."
            )
        # Queued or Running: cancellation confirmed (Req 8.3, 8.4).
        return f"Cancelled job `{job_id}`."

    async def on_targets(self, itx: GateContext) -> str:
        """Handle ``/targets``: list usable Registered_Target names (Req 9.1).

        Gated on the user allowlist; an unauthorized invoker is refused without
        disclosing any names (Req 9.3). When no usable targets exist the reply
        says so (Req 9.2).
        """
        gate = self._gate(itx, admin=False)
        if not gate.allowed:
            return gate.message or _USER_DENIED_MESSAGE

        names = self._require_registry().names()
        if not names:
            return "No targets are configured."
        return "Registered targets: " + ", ".join(names)

    async def on_addtarget(
        self,
        itx: GateContext,
        name: str,
        path: str,
        repo_remote: str = "",
        credentials_ref: str = "",
    ) -> str:
        """Handle ``/addtarget``: register a new target (admin-gated, Req 10.1).

        Gated on the *admin* allowlist; a non-admin invoker is refused (Req 10.2).
        The name must contain a non-whitespace character and be at most
        :data:`~discord_ollama_agent.models.target.TARGET_NAME_MAX_LENGTH`
        Unicode characters (Req 10.6, Property 26). Registration is rejected,
        leaving the registry unchanged, when the name is already in use
        (Req 10.5, Property 25), the path resolves outside the workspace
        (Req 10.3, Property 12), or the path is not an existing directory
        (Req 10.4). On success the target is persisted and the reply references
        the name.
        """
        gate = self._gate(itx, admin=True)
        if not gate.allowed:
            return gate.message or _ADMIN_DENIED_MESSAGE

        # Validate the name up front so a blank/over-length name yields the
        # invalid-name reply rather than a generic validation error (Req 10.6).
        if not name.strip() or len(name) > TARGET_NAME_MAX_LENGTH:
            return _INVALID_TARGET_NAME_MESSAGE

        try:
            candidate = RegisteredTarget(
                name=name,
                directory_path=Path(path),
                repo_remote=repo_remote,
                credentials_ref=credentials_ref,
            )
        except ValidationError:
            return _INVALID_TARGET_NAME_MESSAGE

        registry = self._require_registry()
        try:
            registry.add(candidate)
        except DuplicateTargetError:
            return f"The target name {name!r} is already in use."
        except PathOutsideWorkspaceError:
            return "The target path must be within the workspace."
        except TargetDirectoryNotFoundError:
            return "The target directory does not exist."

        return f"Registered target {candidate.name!r}."

    async def on_revise(self, itx: GateContext, job_id: str, feedback: str) -> str:
        """Handle ``/revise``: continue a Succeeded Job on the same branch.

        Gated on the user allowlist (Req 11.5). A revision is created **iff** the
        feedback has a non-whitespace character (Req 11.8, Property 2), the
        referenced Job exists (Req 11.6, Property 21), its status is Succeeded
        (Req 11.7), and it still retains its branch information (Req 11.9); these
        are exactly the conditions of Property 31. On success a new Queued Job is
        created that continues the referenced Job's branch and the reply carries
        the new identifier.
        """
        gate = self._gate(itx, admin=False)
        if not gate.allowed:
            return gate.message or _USER_DENIED_MESSAGE

        if not feedback.strip():
            return _EMPTY_FEEDBACK_MESSAGE

        manager = self._require_manager()
        base = manager.get(job_id)
        if base is None:
            return _not_found_message(job_id)
        if base.status is not JobStatus.SUCCEEDED:
            return "Only successfully completed jobs can be revised."
        if base.branch_name is None:
            return f"Job `{job_id}` can no longer be revised."

        revision = manager.create_revision(
            base, feedback, channel_id=itx.channel_id, user_id=itx.user_id
        )
        manager.notify()
        return f"Created revision `{revision.id}` (Queued) of job `{job_id}`."

    # -- Lifecycle event sink --------------------------------------------------

    async def on_job_event(self, event: JobEvent) -> None:
        """Render and post a lifecycle status message for ``event`` (Req 6.1-6.3).

        Posts a started/succeeded/failed message to the Job's originating channel
        (Req 6.1, 6.2, 6.3, 11.11). Statuses that have no Discord message
        (Queued, Cancelled) are ignored. Delivery goes through the injected
        :class:`MessageSink` and is retried up to :data:`_MAX_DELIVERY_RETRIES`
        times on failure; if every attempt fails the Discord_Bot records a
        delivery-failure reason in the operator log and makes no further attempts
        (Req 6.7, Property 22).
        """
        content = _render_event(event)
        if content is None:
            return
        await self._deliver(event.channel_id, content, event.job_id)

    async def _deliver(self, channel_id: int, content: str, job_id: str) -> bool:
        """Post ``content`` to ``channel_id``, retrying a bounded number of times.

        Makes one initial attempt plus at most :data:`_MAX_DELIVERY_RETRIES`
        retries (four attempts in all). Returns ``True`` on the first successful
        delivery. If every attempt raises, a delivery-failure reason is logged
        for the operator and ``False`` is returned; no further attempts are made
        (Req 6.7, Property 22).
        """
        sink = self._require_sink()
        last_error: Exception | None = None
        for attempt in range(_MAX_DELIVERY_RETRIES + 1):
            try:
                await sink.send(channel_id, content)
                return True
            except Exception as exc:  # noqa: BLE001 - any delivery error retries
                last_error = exc
        logger.error(
            "Delivery failure: gave up posting status for job %s to channel %s "
            "after %d attempts: %s",
            job_id,
            channel_id,
            _MAX_DELIVERY_RETRIES + 1,
            last_error,
        )
        return False

    # -- Dependency accessors --------------------------------------------------

    def _require_manager(self) -> JobManager:
        """Return the wired :class:`JobManager` or raise if none was provided."""
        if self._job_manager is None:
            raise RuntimeError("DiscordBot requires a JobManager for this command")
        return self._job_manager

    def _require_registry(self) -> TargetRegistry:
        """Return the wired :class:`TargetRegistry` or raise if none was provided."""
        if self._target_registry is None:
            raise RuntimeError("DiscordBot requires a TargetRegistry for this command")
        return self._target_registry

    def _require_sink(self) -> MessageSink:
        """Return the wired :class:`MessageSink` or raise if none was provided."""
        if self._message_sink is None:
            raise RuntimeError("DiscordBot requires a MessageSink to deliver messages")
        return self._message_sink
