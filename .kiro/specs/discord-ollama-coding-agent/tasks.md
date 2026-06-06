# Implementation Plan: Discord-Ollama Coding Agent

## Overview

This plan builds the single-container Python agent incrementally: data models and configuration
first, then the stateless execution services (Ollama client, sandbox, Git), then the orchestration
layer (Coding_Agent, Job_Manager), then the Discord transport, and finally the startup wiring,
container packaging, and sample configuration.

Each correctness property from the design (Properties 1–31) is implemented as a single Hypothesis
property-based test placed next to the code that establishes the behavior. Every property test must:
run a **minimum of 100 iterations** (`max_examples >= 100`), mock external dependencies (in-memory
fake `OllamaClient`, temp-dir local git repo, stub command runner), and carry a tag comment in the
format **`Feature: discord-ollama-coding-agent, Property {n}: {property_text}`**. Example, integration,
and smoke tests follow the design's Testing Strategy.

The implementation language is Python (per the design). Recommended libraries: discord.py, httpx,
pydantic v2, GitPython + git subprocess, asyncio subprocess, Hypothesis + pytest (+ pytest-asyncio).

## Tasks

- [x] 1. Set up project scaffolding and shared error types
  - [x] 1.1 Create the Python project structure and dependencies
    - Create `pyproject.toml` declaring runtime deps (discord.py, httpx, pydantic v2, pydantic-settings, PyYAML, GitPython) and dev deps (pytest, pytest-asyncio, hypothesis)
    - Create the `discord_ollama_agent/` package with `__init__.py` and the empty module layout: `models/`, `config_loader.py`, `target_registry.py`, `ollama_client.py`, `execution_sandbox.py`, `git_manager.py`, `coding_agent.py`, `job_manager.py`, `discord_bot.py`, `main.py`
    - Create `discord_ollama_agent/errors.py` defining the exception hierarchy used throughout: `StartupError`, `SandboxInitError`, `PathEscapeError`, `WriteError`, `GenerationTimeout`, `OllamaError`
    - Create the `tests/` package and configure pytest (`pytest.ini`/`pyproject` `[tool.pytest.ini_options]`) with `pytest-asyncio` mode and Hypothesis default settings
    - _Requirements: (foundational; supports all)_

- [x] 2. Implement core data models with validation bounds
  - [x] 2.1 Implement the Generation models
    - In `models/generation.py` implement `FileEntry` (relative `path`, full `content`), `GenerationResult` (`files: list[FileEntry]`, zero or more), and `GenerationRequest` (`idea`, `context_files`, `prior_error`, `feedback`) as pydantic v2 models
    - _Requirements: 3.3, 3.7, 3.12, 4.1, 4.2_

  - [x] 2.2 Write property test for Generation_Result serialization round-trip
    - **Property 27: Generation_Result round-trips through serialization** — serialize a valid `GenerationResult` to the structured wire format and parse it back, asserting an equivalent result (same ordered file entries, identical paths and content)
    - Tag: `Feature: discord-ollama-coding-agent, Property 27`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 3.7**

  - [x] 2.3 Implement the Job model and lifecycle types
    - In `models/job.py` implement `JobStatus` enum (Queued/Running/Succeeded/Failed/Cancelled) and `Job` (id, target_name, idea, status, submitted_at, channel_id, user_id, attempts_completed, branch_name, failure_reason, result_note, is_revision, base_job_id)
    - Define the `JobEvent` payload type emitted on transitions (job id, new status, branch/result note, failure reason) for the Discord event sink
    - _Requirements: 6.1, 6.2, 6.3, 7.1, 7.4, 11.1, 11.9_

  - [x] 2.4 Implement the RegisteredTarget model
    - In `models/target.py` implement `RegisteredTarget` (name, directory_path, optional build_command, repo_remote, default_branch, branch_scheme, credentials_ref) with name validation (1..64 non-whitespace chars)
    - _Requirements: 5.1, 5.3, 5.4, 5.7, 10.6, 12.1_

  - [x] 2.5 Implement the SystemConfig models with fail-fast bounds
    - In `models/config.py` implement `ContextBudget` (max_file_count 1..1000, max_total_bytes 1024..67108864), `OllamaConfig` (endpoint_url as http/https, non-empty model, generation_timeout_s 30..600), and `SystemConfig` (workspace_dir, max_build_attempts 1..10, concurrency_limit 1..100, execution_timeout_s 1..3600, context_budget, non-empty allowed_channels, authorized_users, admin_users, default_target, max_idea_length, push_timeout_s, build_output_cap_bytes default 1 MiB)
    - Enforce all bounds at construction with pydantic validators so violations raise with a message identifying the offending value
    - _Requirements: 1.2, 1.7, 1.8, 2.1, 2.2, 2.5, 3.10, 4.6, 4.8, 4.9, 4.14, 5.5, 7.6, 13.4, 13.5_

- [x] 3. Implement the Config_Loader (fail-fast startup validation)
  - [x] 3.1 Implement ConfigLoader.load
    - In `config_loader.py` implement `ConfigLoader.load(path)` reading YAML and constructing `SystemConfig`, translating validation failures into `StartupError` with a message identifying the offending value (Ollama endpoint/model, URL scheme, max_build_attempts, concurrency_limit, context budget, allowed_channels)
    - _Requirements: 2.1, 2.2, 2.5, 4.9, 4.14, 7.6, 13.4, 13.5_

  - [x] 3.2 Write property test for startup configuration validation
    - **Property 8: Startup configuration validation accepts exactly in-bounds values** — generate config values across and outside the bounds and assert startup succeeds iff `max_build_attempts` ∈ [1,10], `concurrency_limit` ∈ [1,100], context-budget file count ∈ [1,1000], context-budget bytes ∈ [1024,67108864], and `allowed_channels` is non-empty; otherwise a `StartupError` identifying the value is raised
    - Tag: `Feature: discord-ollama-coding-agent, Property 8`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 4.9, 4.14, 7.6, 13.5**

  - [x] 3.3 Write example/edge tests for Ollama config validation
    - Missing/empty Ollama endpoint or model raises `StartupError` (2.2); endpoint present but not a valid http/https URL raises `StartupError` (2.5)
    - _Requirements: 2.2, 2.5_

  - [x] 3.4 Write smoke test for config loading
    - Single execution: a valid config file is read and the Ollama endpoint/model and Allowed_Channels are populated on `SystemConfig`
    - _Requirements: 2.1, 13.4_

- [x] 4. Implement the Target_Registry (load/partition, resolve, persist)
  - [x] 4.1 Implement TargetRegistry
    - In `target_registry.py` implement `load` (partition into usable + excluded-with-reason, logging excluded name + missing value), `get`, `names`, `add` (name + workspace-containment validation, atomic write-to-temp + `os.replace`), and `exists`
    - Implement workspace containment via `Path.resolve()` on workspace and candidate, checking the resolved candidate is relative to the resolved workspace (rejecting `..` and resolved symlinks)
    - _Requirements: 1.6, 9.1, 9.2, 10.1, 10.3, 10.4, 10.5, 10.6, 12.1, 12.2, 12.3, 12.4_

  - [x] 4.2 Write property test for workspace-confined registration
    - **Property 12: Target registration is confined to the Workspace_Directory** — generate in-workspace and escaping paths (absolute, `..`, symlink-backed) and assert a target is registered iff its path resolves inside the workspace
    - Tag: `Feature: discord-ollama-coding-agent, Property 12`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 10.3**

  - [x] 4.3 Write property test for registry partition and unregistered-build rejection
    - **Property 23: Registry validation partitions targets and unregistered builds are rejected** — generate registries with valid/invalid targets and assert a target is usable iff its in-workspace path and remote are present, excluded targets are logged, and lookups for non-usable names are rejected as not-registered
    - Tag: `Feature: discord-ollama-coding-agent, Property 23`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 1.6, 12.2, 12.3, 12.4**

  - [x] 4.4 Write property test for registration persistence across reloads
    - **Property 24: Target registration persists across reloads** — add a valid target, reload the registry from disk, and assert the added target is present with identical configuration
    - Tag: `Feature: discord-ollama-coding-agent, Property 24`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 10.1**

  - [x] 4.5 Write property test for duplicate-name rejection
    - **Property 25: Duplicate target names are rejected without clobbering** — for a name matching an existing target, assert registration is rejected and the existing target's configuration is unchanged
    - Tag: `Feature: discord-ollama-coding-agent, Property 25`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 10.5**

  - [x] 4.6 Write property test for target name validation
    - **Property 26: Target name validation enforces length and non-whitespace** — generate Unicode names (including whitespace-only and >64 chars) and assert acceptance iff the name has at least one non-whitespace character and ≤64 Unicode characters
    - Tag: `Feature: discord-ollama-coding-agent, Property 26`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 10.6**

  - [x] 4.7 Write example tests for registration path existence
    - Nonexistent path and a path that is not a directory are rejected with the appropriate reply reason
    - _Requirements: 10.4_

- [x] 5. Implement the Execution_Sandbox (path confinement + subprocess)
  - [x] 5.1 Implement ExecutionSandbox
    - In `execution_sandbox.py` implement `init_working_dir(job, source)` (distinct per-Job root, raising `SandboxInitError`), `resolve_within` (join + `resolve()` + relative-to-root check, raising `PathEscapeError`), `write_file` (path-validated whole-file write, raising `WriteError`), and `run_command` (asyncio subprocess in the working dir, new process group, output capture capped to `build_output_cap_bytes` (1 MiB), termination on timeout or cancel)
    - _Requirements: 1.9, 4.1, 4.2, 4.3, 4.6, 4.8, 4.10, 4.11, 7.2, 7.7_

  - [x] 5.2 Write property test for sandbox write confinement
    - **Property 11: Sandbox confines all writes to the Job working directory** — generate in-bounds relative paths and escaping paths (absolute, `..`, symlink-backed) and assert a write is permitted iff it resolves inside the working dir, with no file outside the dir modified on denial
    - Tag: `Feature: discord-ollama-coding-agent, Property 11`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 4.2**

  - [x] 5.3 Write property test for file-write round-trip within the sandbox
    - **Property 13: File writes round-trip within the sandbox** — for sets of in-bounds file entries, assert reading each written path back yields exactly the content written, with later writes overwriting earlier content
    - Tag: `Feature: discord-ollama-coding-agent, Property 13`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 4.1**

  - [x] 5.4 Write property test for build-output capping
    - **Property 15: Captured build output is capped at 1 MiB** — feed the stub command runner outputs of varying sizes and assert captured output never exceeds 1,048,576 bytes
    - Tag: `Feature: discord-ollama-coding-agent, Property 15`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 4.6**

  - [x] 5.5 Write property test for per-Job working-copy isolation
    - **Property 5: Working copies are isolated per Job** — for sets of concurrently tracked Jobs (including multiple Jobs against the same target), assert each Job's working-copy root is distinct from every other Job's
    - Tag: `Feature: discord-ollama-coding-agent, Property 5`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 1.9, 7.2, 7.7**

  - [x] 5.6 Write integration tests for command execution
    - Build command actually runs in the sandbox working directory with the correct cwd (4.3); a sleeping command exceeding the execution timeout is terminated (4.8)
    - _Requirements: 4.3, 4.8_

  - [x] 5.7 Write example tests for sandbox failure paths
    - Forced sandbox-init failure surfaces `SandboxInitError` mapping to a sandbox-initialization-failure reason (4.10); a forced write failure surfaces `WriteError` mapping to a write-failure reason (4.11)
    - _Requirements: 4.10, 4.11_

- [x] 6. Implement the Ollama_Client (connectivity + generation)
  - [x] 6.1 Implement OllamaClient
    - In `ollama_client.py` implement `check_connectivity` (GET `{endpoint}/api/tags`, confirm configured model present within 10s) and `generate` (POST `{endpoint}/api/chat` with `format` = `GenerationResult.model_json_schema()`, per-request generation timeout raising `GenerationTimeout`, error responses raising `OllamaError`), sending only to the configured endpoint
    - _Requirements: 2.3, 2.4, 3.3, 3.6, 3.10, 3.11_

  - [x] 6.2 Write property test for single-endpoint egress
    - **Property 29: Generation traffic targets only the configured endpoint** — for any generation request, assert (via a recording httpx transport/mock) the only destination contacted is the configured `Ollama_Endpoint` and no idea/generated content goes elsewhere
    - Tag: `Feature: discord-ollama-coding-agent, Property 29`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 3.6**

  - [x] 6.3 Write example test for structured-output schema
    - Assert the generation request carries the `GenerationResult` JSON schema in the Ollama `format` field
    - _Requirements: 3.3_

  - [x] 6.4 Write integration tests for the connectivity check
    - Against a mocked `/api/tags`: model present confirms readiness; model absent / unreachable within 10s flips degraded mode; a later check confirming the model recovers readiness
    - _Requirements: 2.3, 2.4_

  - [x] 6.5 Write integration tests for generation timeout and error mapping
    - A delayed mock response triggers `GenerationTimeout` → timeout reason (3.10); a model error response maps to `OllamaError` → recorded error reason (3.11)
    - _Requirements: 3.10, 3.11_

- [x] 7. Implement the Git_Manager (sync, branch, commit, push)
  - [x] 7.1 Implement GitManager
    - In `git_manager.py` implement `sync_default_branch` (fetch + reset working copy to the target's default branch, mapping failure to a sync-failure reason), `branch_name_for` (derive from Job id via the target's branch scheme; includes target name), `commit_and_push` (no-changes → Succeeded note; else create branch from synced state, stage add/modify/delete, commit referencing the Job id, push with per-target credentials within the push timeout; missing credentials → no push; refuse to run for a Cancelled Job), and `commit_and_push_revision` (add commits to the existing branch and push)
    - Resolve credentials at runtime from the `credentials_ref` (env var/secret path), injecting via per-invocation environment/credential helper; never persist or log secret values
    - Run blocking GitPython calls via `asyncio.to_thread`; apply timeouts to fetch/push
    - _Requirements: 4.12, 4.13, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 11.4_

  - [x] 7.2 Write property test for branch/commit identifier references
    - **Property 19: Branch and commit reference the target and Job identifier** — for any Job, assert the derived branch name contains the target name and the commit message references the Job id
    - Tag: `Feature: discord-ollama-coding-agent, Property 19`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 5.1, 5.2**

  - [x] 7.3 Write property test for the no-changes outcome
    - **Property 18: No file changes yields Succeeded with a no-changes note and no push** — for a working copy with no diff from the synced state (temp-dir local repo), assert the Job transitions to Succeeded with a no-changes note and commit/push are skipped
    - Tag: `Feature: discord-ollama-coding-agent, Property 18`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 5.6**

  - [x] 7.4 Write property test for missing-credentials guard
    - **Property 17: Missing credentials prevent any push** — for any target whose credentials reference resolves to nothing, assert the Job fails with missing-credentials and no push is attempted
    - Tag: `Feature: discord-ollama-coding-agent, Property 17`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 5.7**

  - [x] 7.5 Write property test for the cancelled-Job guard
    - **Property 16: Cancelled Jobs never stage, commit, or push** — for any Job in/transitioning to Cancelled, assert the Git_Manager performs no staging, commit, or push regardless of files produced
    - Tag: `Feature: discord-ollama-coding-agent, Property 16`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 5.9, 8.4, 8.5**

  - [x] 7.6 Write integration tests for default-branch sync
    - Sync from a local fixture remote updates the working copy to the default branch (4.12); a failing fetch/reset maps to a sync-failure reason (4.13)
    - _Requirements: 4.12, 4.13_

  - [x] 7.7 Write integration tests for commit and push
    - Push a new branch to a fixture remote with per-target credentials resolved from a reference, transitioning to Succeeded (5.3, 5.4, 5.8); push failure/timeout maps to a push-error reason (5.5)
    - _Requirements: 5.3, 5.4, 5.5, 5.8_

  - [x] 7.8 Write integration test for revision push
    - A revision adds new commits to the referenced Job's existing branch on the fixture remote and pushes them
    - _Requirements: 11.4_

- [x] 8. Checkpoint - stateless services
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement the Coding_Agent (context build + iterative loop)
  - [x] 9.1 Implement context construction
    - In `coding_agent.py` implement `_build_context` (empty working copy → no context; non-empty → budget-bounded subset within max file count and max total bytes) and `_build_revision_context` (load referenced branch contents + feedback into a `GenerationRequest`)
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 11.2_

  - [x] 9.2 Write property test for budget-bounded context selection
    - **Property 9: Context selection stays within the configured budget** — for any file set and valid budget, assert the selected context has at most the max file count and at most the max total bytes, and generation proceeds with that subset
    - Tag: `Feature: discord-ollama-coding-agent, Property 9`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 3.4, 3.5**

  - [x] 9.3 Write property test for context presence vs. emptiness
    - **Property 10: Context presence follows working-copy emptiness** — assert the first generation request includes existing-file context iff the working copy is non-empty
    - Tag: `Feature: discord-ollama-coding-agent, Property 10`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 3.1, 3.2**

  - [x] 9.4 Implement the run loop and failure handling
    - Implement `CodingAgent.run(job, cancel)` orchestrating sync → sandbox init → context build → generate → parse → write → build → fix loop → commit/push, parsing completions into `GenerationResult`, feeding parse errors/build output back as retries, checking the `CancellationToken` between phases, and recording terminal failure reasons (invalid-output, empty-generation, timeout, error, write-failure, max-attempts-exhausted, sync-failure, sandbox-init-failure); emit `JobEvent`s on transitions
    - _Requirements: 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 4.1, 4.4, 4.5, 4.6, 4.7, 4.10, 4.11, 8.5, 11.3_

  - [x] 9.5 Write property test for attempt-bounded retry with feedback
    - **Property 14: Retry is attempt-bounded and feeds prior errors back** — for sequences of failing attempts (malformed result or non-zero build), assert a new attempt begins carrying the prior error while completed attempts < max, the Job fails with the matching reason at max, and total attempts never exceed `Maximum_Build_Attempts`
    - Tag: `Feature: discord-ollama-coding-agent, Property 14`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 3.8, 3.9, 4.5, 4.6**

  - [x] 9.6 Write property test for empty-generation handling
    - **Property 28: Zero-file generations fail as empty-generation** — assert a completion parsing to zero file entries fails with empty-generation, and any result with ≥1 file entry proceeds to the write step
    - Tag: `Feature: discord-ollama-coding-agent, Property 28`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 3.12**

  - [x] 9.7 Write example tests for build-branch decisions
    - Build exit 0 proceeds to the commit step (4.4); a target with no Build_Command skips the build and proceeds to commit (4.7)
    - _Requirements: 4.4, 4.7_

- [x] 10. Implement the Job_Manager (identity, queue, dispatch, cancel)
  - [x] 10.1 Implement Job identity and the in-memory table
    - In `job_manager.py` implement `create_job` (id `"{target_name}-{unique_suffix}"` via monotonic counter/UUID, status Queued, enqueue), `create_revision` (new Job continuing `base_job.branch_name`), and `get`; ensure ids are unique and never reused
    - _Requirements: 1.1, 1.3, 7.1, 11.1, 11.10_

  - [x] 10.2 Write property test for Job identity
    - **Property 4: Job identity contains the target name, is unique, and is never reused** — for any sequence of creations (including after terminal transitions), assert every id contains its target name and all ids are pairwise unique with no reuse
    - Tag: `Feature: discord-ollama-coding-agent, Property 4`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 1.3, 7.1**

  - [x] 10.3 Implement the dispatcher, semaphore, and cancellation
    - Implement the FIFO `asyncio.Queue`, the `asyncio.Semaphore` sized to the concurrency limit, `_dispatch_loop` (pop earliest-submission Queued Job → Running → spawn worker `asyncio.Task`, freeing a slot on every terminal transition), and `cancel` (Queued → remove + Cancelled; Running → signal `CancellationToken` + terminate build + Cancelled; terminal → reject)
    - _Requirements: 7.3, 7.4, 7.5, 7.7, 8.3, 8.4, 8.6_

  - [x] 10.4 Write property test for the concurrency invariant
    - **Property 6: Running Jobs never exceed the concurrency limit** — use a Hypothesis `RuleBasedStateMachine` interleaving submissions and completions and assert Running count never exceeds the limit and every terminal transition frees exactly one slot
    - Tag: `Feature: discord-ollama-coding-agent, Property 6`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 7.3, 7.5, 7.7**

  - [x] 10.5 Write property test for FIFO dispatch order
    - **Property 7: Dispatch order is FIFO by submission time** — for any set of Queued Jobs, assert the Job promoted when a slot frees is the one with the earliest submission time
    - Tag: `Feature: discord-ollama-coding-agent, Property 7`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 7.4**

  - [x] 10.6 Write property test for cancellation outcome
    - **Property 30: Cancellation outcome depends only on current status** — assert `/cancel` removes-and-Cancels a Queued Job, signals-stop/terminates-and-Cancels a Running Job, and is rejected (status unchanged) for terminal Jobs
    - Tag: `Feature: discord-ollama-coding-agent, Property 30`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 8.3, 8.6**

  - [x] 10.7 Write property test for revision eligibility and same-branch continuation
    - **Property 31: Revision eligibility and same-branch continuation** — assert a revision is created iff the referenced Job exists, is Succeeded, retains its branch, and feedback is non-whitespace; when created it continues on the referenced branch and loads that branch's contents plus feedback as context; otherwise the appropriate rejection is returned with no revision
    - Tag: `Feature: discord-ollama-coding-agent, Property 31`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 11.1, 11.2, 11.7, 11.9**

- [x] 11. Checkpoint - orchestration
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement the Discord_Bot (transport, gating, commands, status)
  - [x] 12.1 Implement command gating
    - In `discord_bot.py` implement `_gate(itx, *, admin)` enforcing the conjunction: channel ∈ Allowed_Channels AND user ∈ (admin allowlist for `/addtarget`, else user allowlist); both must pass for a command to be acted upon, otherwise produce a not-permitted/denial reply with no state change
    - _Requirements: 1.4, 6.6, 8.1, 9.3, 10.2, 11.5, 13.1, 13.2, 13.3_

  - [x] 12.2 Write property test for the gating conjunction
    - **Property 1: Command gating requires both channel and user authorization** — for any command, user, and channel, assert the command is acted upon iff channel is allowed AND user is authorized for that command; otherwise no state change and a denial reply
    - Tag: `Feature: discord-ollama-coding-agent, Property 1`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 1.4, 6.6, 8.1, 9.3, 10.2, 11.5, 13.1, 13.2, 13.3**

  - [x] 12.3 Implement the six slash commands and the status event sink
    - Register and implement `/build`, `/status`, `/cancel`, `/targets`, `/addtarget`, `/revise` with argument validation (non-whitespace idea/feedback, idea length cap, target resolution, default-target routing) and message rendering; implement `on_job_event` rendering started/succeeded/failed messages with delivery retry up to 3 times then a logged delivery-failure reason; all replies within the 5-second window
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 6.1, 6.2, 6.3, 6.4, 6.5, 6.7, 8.2, 9.1, 9.2, 10.1, 11.6, 11.11_

  - [x] 12.4 Write property test for whitespace-only input rejection
    - **Property 2: Whitespace-only required inputs are rejected** — for any all-whitespace `/build` idea or `/revise` feedback, assert no Job/revision is created and a non-empty-input reply is produced
    - Tag: `Feature: discord-ollama-coding-agent, Property 2`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 1.5, 11.8**

  - [x] 12.5 Write property test for the idea length cap
    - **Property 3: Idea length cap is enforced** — for any configured max and idea string, assert a Job is created iff the idea's Unicode length ≤ max; otherwise a too-long reply and no Job
    - Tag: `Feature: discord-ollama-coding-agent, Property 3`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 1.8**

  - [x] 12.6 Write property test for status message content
    - **Property 20: Status messages carry the Job identifier and correct state information** — assert started/succeeded/failed messages and the `/status` reply contain the Job id and matching state (branch name or no-changes note for Succeeded, failure reason for Failed, current status for `/status`)
    - Tag: `Feature: discord-ollama-coding-agent, Property 20`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 6.2, 6.3, 6.4, 11.11**

  - [x] 12.7 Write property test for not-found replies
    - **Property 21: Not-found replies reference the queried identifier** — for `/status`, `/cancel`, or `/revise` against an unknown Job id, assert the reply is a not-found message including the queried id and no state changes occur
    - Tag: `Feature: discord-ollama-coding-agent, Property 21`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 6.5, 8.2, 11.6**

  - [x] 12.8 Write property test for bounded status-delivery retries
    - **Property 22: Status delivery retries are bounded** — for any sequence of delivery failures, assert at most 3 retries are attempted, then a delivery-failure reason is logged and no further attempts are made
    - Tag: `Feature: discord-ollama-coding-agent, Property 22`; Hypothesis `max_examples >= 100`
    - **Validates: Requirements 6.7**

  - [x] 12.9 Write example test for the empty-registry listing
    - `/targets` with no Registered_Targets replies with a no-targets-configured message
    - _Requirements: 9.2_

  - [x] 12.10 Write integration test for the started message timing
    - On a Job's transition to Running, the started message referencing the Job id is posted to the originating channel within the timing budget
    - _Requirements: 6.1_

  - [x] 12.11 Write example test for omitted target without a default
    - `/build` with an omitted target and no Default_Target configured creates no Job and replies that a target must be specified
    - _Requirements: 1.7_

- [x] 13. Wire the system together (startup sequence)
  - [x] 13.1 Implement the main entry point and startup sequence
    - In `main.py` implement: load + validate system config (exit non-zero on `StartupError`), load the persisted registry (exclude invalid targets with logs), run the Ollama connectivity check (set degraded mode rejecting new Jobs with an unavailable-model message until a later check confirms), construct and wire `ExecutionSandbox`, `GitManager`, `OllamaClient`, `CodingAgent`, `JobManager` (dispatcher loop), and `DiscordBot` (gateway connect + command registration + JobEvent sink)
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 4.9, 4.14, 7.6, 12.1, 12.2, 13.4, 13.5_

  - [x] 13.2 Write integration test for startup wiring
    - With a mocked Discord gateway and mocked Ollama, assert a valid config + registry reaches a ready state with commands registered and the dispatcher running; an invalid config exits without a ready state; a failed connectivity check enters degraded mode
    - _Requirements: 2.4, 12.1_

- [x] 14. Package the container and provide samples
  - [x] 14.1 Create the container build
    - Create a `Dockerfile` (Python base, install the package and dependencies including `git`, set the entry point to `main.py`) and a `.dockerignore`
    - _Requirements: (single-container host-isolation boundary)_

  - [x] 14.2 Create sample configuration and a sample Target_Registry
    - Add a documented sample system config YAML (Ollama endpoint/model, workspace_dir, bounds, allowed_channels, allowlists, default_target) and a sample `Target_Registry` JSON with one or more example targets using credential references (never secret values)
    - _Requirements: 2.1, 10.1, 12.1, 13.4_

- [x] 15. Final checkpoint - full test suite
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks (property, example, integration, smoke) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each property sub-task implements exactly one design Property (1–31), runs Hypothesis with `max_examples >= 100`, mocks external dependencies (fake Ollama client, temp-dir local git repo, stub command runner), and carries the `Feature: discord-ollama-coding-agent, Property {n}` tag comment.
- Example/integration/smoke tests follow the design's Testing Strategy for criteria classified EXAMPLE, EDGE_CASE, INTEGRATION, or SMOKE.
- Credentials are always handled as references; secret values are never persisted or logged.
- Checkpoints provide incremental validation at the boundaries between stateless services, orchestration, and transport.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.3", "2.4", "2.5", "14.1", "14.2"] },
    { "id": 2, "tasks": ["2.2", "3.1", "4.1", "5.1", "6.1", "7.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "6.2", "6.3", "6.4", "6.5", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "9.1", "10.1"] },
    { "id": 4, "tasks": ["9.2", "9.3", "9.4", "10.2", "12.1"] },
    { "id": 5, "tasks": ["9.5", "9.6", "9.7", "10.3", "12.2"] },
    { "id": 6, "tasks": ["10.4", "10.5", "10.6", "10.7", "12.3"] },
    { "id": 7, "tasks": ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "12.10", "12.11", "13.1"] },
    { "id": 8, "tasks": ["13.2"] }
  ]
}
```
