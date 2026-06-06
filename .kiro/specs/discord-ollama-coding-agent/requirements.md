# Requirements Document

## Introduction

This feature defines a workspace automation system that turns coding ideas submitted through Discord into committed code changes using a locally hosted Ollama language model. An Authorized_User runs a Discord slash command with a target name and an idea, an autonomous Coding_Agent iteratively generates code, writes files, and runs the build command configured for that target, and the resulting changes are pushed to that target's configured Git repository. Status updates are reported back to the originating Discord channel so the user knows when work starts, succeeds, or fails. Authorized_Users may also list the available targets, revise successfully completed work, and query or cancel Jobs from Discord.

The System is implemented in Python. The Discord_Bot and Coding_Agent are developed as a single reusable agent project that is not tied to any one codebase. The agent operates on a set of Registered_Targets rather than a single hardcoded codebase. The operator configures a Workspace_Directory, a parent folder under which all valid target directories must reside, and seeds an initial base list of Registered_Targets. Each Registered_Target carries its own configuration: a unique name, a directory path that resolves inside the Workspace_Directory, an optional Build_Command, an associated Target_Repository with its own Git remote, its own branch-naming scheme, and its own Git credentials reference. A Registered_Target may contain an existing codebase that the agent modifies, or it may be empty so the agent writes fresh code. Because each Registered_Target has its own build command, Git remote, branch-naming scheme, and Git credentials, different targets may build and push differently, enabling parallel work across heterogeneous projects (for example, a Python backend and a JavaScript frontend held in separate repositories). Only the Ollama_Endpoint URL and model remain global across all targets. Admin_Users can register additional targets from Discord, and the full set of Registered_Targets is persisted in a Target_Registry that survives Container restarts. When a `/build` command omits a target, the System uses a configured default Registered_Target. The agent code remains generic across different target projects: targets are selected through configuration and the Target_Registry rather than hardcoded.

The entire System runs inside a single container, and the container is the host-isolation boundary between the System and the host machine. The agent does not create a separate container per Job; instead, generated and executed code is confined to a per-Job working directory inside the container, with path-traversal denied and execution subject to timeouts and resource bounds.

Regarding network access (non-functional / security framing): the Container provides outbound network access because build and test commands must be able to fetch dependencies (for example, package installs during a build). This is an accepted risk. Model-generated build and test commands run with outbound network access available, and the Container, not network egress filtering, is the host-isolation boundary. The primary controls against misuse are the user allowlist, the admin allowlist, and the configured set of Allowed_Channels that gate which commands are accepted and from whom; outbound network access for builds is not denied.

The system is intended for single-tenant or small-team use where the Ollama model runs on local infrastructure (no third-party LLM API). When a target already contains a codebase, the agent supplies a budget-bounded selection of the existing project files to the model as context so it modifies real code rather than guessing, refreshes the target from its remote before each Job so work is based on current code, and accepts commands only in operator-configured Discord channels. The design emphasizes safe execution of generated code, an iterative build-and-fix loop, reliable Git operations, and clear feedback for long-running jobs.

## Glossary

- **System**: The complete Discord-to-Ollama coding automation workspace, comprising the Discord Bot, the Coding Agent, the Ollama Client, the Execution Sandbox, and the Git Manager, all running inside a single Container.
- **Container**: The single OS-level container in which the entire System runs; the Container is the host-isolation boundary between the System and the host machine.
- **Discord_Bot**: The component that connects to Discord, receives slash commands, and posts responses back to a Discord channel.
- **Coding_Agent**: The autonomous component that interprets a coding idea, orchestrates code generation through the Ollama_Client, writes changes into a Job's working copy of the Target_Project_Directory, runs the Build_Command configured for the Job's Registered_Target, and iteratively feeds build error output back to the Ollama model to revise the changes until the build succeeds or the configured Maximum_Build_Attempts is reached.
- **Ollama_Client**: The component that sends prompts to and receives completions from a locally hosted Ollama model server over a configured HTTP endpoint.
- **Generation_Result**: The structured output returned by the Ollama model for a single generation attempt, consisting of zero or more file entries, where each file entry comprises a relative file path and the full intended content of the file at that path. The Coding_Agent writes each file entry as a whole-file write (creating or overwriting the file), not as a diff or patch.
- **Context_Budget**: The configured maximum amount of existing project code the Coding_Agent includes in a generation request, expressed as a maximum number of files and a maximum total size in bytes; the budget bounds context inclusion so a generation request stays within the model's context-window limits.
- **Execution_Sandbox**: The per-Job working-directory boundary inside the Container within which generated and executed code is written, built, and tested; file operations are confined to the Job's working directory (path traversal denied) and execution is subject to timeouts and resource bounds. The Execution_Sandbox does not create a separate container per Job.
- **Git_Manager**: The component that creates branches, stages changes, commits, and pushes to the Git remote of the Job's Registered_Target's Target_Repository using that target's configured Git credentials.
- **Job**: A single unit of work created from one user idea, tracked in memory from submission through completion or failure. Each Job is identified by a Job identifier composed of the associated Registered_Target's name plus a unique component (for example, the target name combined with a unique suffix); the Job identifier contains the associated target name and is unique among all Jobs created during the System's lifetime and is not reused.
- **Job_Status**: The state of a Job, one of: Queued, Running, Succeeded, Failed, or Cancelled. Succeeded, Failed, and Cancelled are terminal states.
- **Target_Project_Directory**: The directory of the Registered_Target selected for a given Job, located inside the Workspace_Directory, that contains the code the Coding_Agent modifies; the directory may contain an existing codebase to modify or may be empty for new code. The Target_Project_Directory for a Job is selected by target name through configuration and the Target_Registry, and is not hardcoded in the agent. When a `/build` command omits a target name, the configured default Registered_Target is used.
- **Workspace_Directory**: The operator-configured parent folder under which all valid target directories must reside; the Workspace_Directory is the containment boundary used when registering and resolving Registered_Targets.
- **Registered_Target**: A named target directory the System knows about, carrying its own per-target configuration. Each Registered_Target has a unique name; a directory path that resolves to a location inside the Workspace_Directory; its own optional Build_Command; its own associated Target_Repository with its own Git remote, branch-naming scheme or prefix, and Git credentials reference. The set of Registered_Targets consists of an operator-seeded base list plus targets added by Admin_Users.
- **Target_Registry**: The persisted store of Registered_Targets, held in a registry/config file, that survives Container restarts; the operator-seeded list forms the base and Admin_User-added targets are appended persistently.
- **Default_Target**: The Registered_Target, identified in configuration, that the System uses for a `/build` command when no target name is supplied.
- **Target_Repository**: The Git repository associated with a specific Registered_Target, into which the Coding_Agent's changes for Jobs against that target are committed and pushed. Each Registered_Target has its own associated Target_Repository, its own Git remote, and its own Git credentials reference; these are not shared globally across targets.
- **Build_Command**: The command or commands, configured by the operator per Registered_Target in that target's configuration, that the Coding_Agent runs to build and/or test generated changes; the Build_Command used for a Job is the one configured for the Job's Registered_Target. The System does not auto-detect the Build_Command and the Ollama model does not select it.
- **Maximum_Build_Attempts**: The configured maximum number of generate-and-build attempts the Coding_Agent performs for a single Job before recording a max-attempts-exhausted failure; default 3.
- **Ollama_Endpoint**: The configured URL and model name identifying the local Ollama server and model used for generation.
- **Authorized_User**: A Discord user whose identifier is present in the configured allowlist permitted to submit Jobs, revise successfully completed Jobs, list Registered_Targets, query the status of Jobs, and cancel Jobs.
- **Admin_User**: A Discord user whose identifier is present in the configured admin allowlist. The admin allowlist is a separate, additional privilege specifically for registering new Registered_Targets; Admin_Users are the only users permitted to register new targets. Admin allowlist membership is independent of the Authorized_User allowlist.
- **Allowed_Channels**: The operator-configured set of Discord channel identifiers in which the Discord_Bot accepts slash commands and posts responses; commands invoked in channels outside this set are not acted upon.

## Requirements

### Requirement 1: Accept Coding Ideas from Discord

**User Story:** As a developer, I want to submit a coding idea for a chosen target through a Discord slash command, so that I can start an automated build against the right project without leaving Discord.

#### Acceptance Criteria

1. WHEN an Authorized_User invokes the `/build` slash command with a target name argument that matches a Registered_Target and an idea argument containing at least one non-whitespace character, THE Discord_Bot SHALL create a new Job with Job_Status set to Queued, associated with the matched Registered_Target.
2. WHEN an Authorized_User invokes the `/build` slash command with the target name argument omitted, an idea argument containing at least one non-whitespace character, and a Default_Target configured, THE Discord_Bot SHALL create a new Job with Job_Status set to Queued, associated with the Default_Target.
3. WHEN a Job is created, THE Discord_Bot SHALL reply in the originating Discord channel, within 5 seconds of Job creation, with the Job identifier, where that identifier contains the associated Registered_Target's name and is unique across all Jobs tracked by the System.
4. IF a user whose Discord identifier is not present in the configured allowlist invokes the `/build` slash command, THEN THE Discord_Bot SHALL NOT create a Job and SHALL reply, within 5 seconds of invocation, with a message indicating the user is not authorized to submit Jobs.
5. IF the idea argument supplied to the `/build` slash command contains no non-whitespace characters, THEN THE Discord_Bot SHALL NOT create a Job and SHALL reply, within 5 seconds of invocation, with a message requesting a non-empty idea.
6. IF the target name argument supplied to the `/build` slash command does not match any Registered_Target, THEN THE Discord_Bot SHALL NOT create a Job and SHALL reply, within 5 seconds of invocation, with a message indicating the target is not registered.
7. IF the target name argument is omitted AND no Default_Target is configured, THEN THE Discord_Bot SHALL NOT create a Job and SHALL reply, within 5 seconds of invocation, with a message indicating a target must be specified.
8. WHERE a maximum idea length is configured, IF the idea argument's length measured in Unicode characters exceeds the configured maximum, THEN THE Discord_Bot SHALL NOT create a Job and SHALL reply, within 5 seconds of invocation, with a message indicating the idea exceeds the configured maximum character length.
9. WHEN a Job is created, THE Coding_Agent SHALL operate on the associated Registered_Target's directory using an isolated working copy in accordance with Requirement 7.

### Requirement 2: Configure the Ollama Endpoint and Model

**User Story:** As an operator, I want to configure which local Ollama server and model are used, so that I can point the System at my own infrastructure.

#### Acceptance Criteria

1. WHEN the System starts, THE System SHALL read the Ollama_Endpoint URL and model name from configuration.
2. IF the Ollama_Endpoint URL or model name is missing or empty in configuration at startup, THEN THE System SHALL terminate startup without entering a ready state and SHALL emit an error message identifying the missing configuration value.
3. WHEN the System starts and the configured Ollama_Endpoint URL and model name are present and valid, THE Ollama_Client SHALL perform a connectivity check that confirms the configured model is available at the Ollama_Endpoint.
4. IF the connectivity check does not confirm the configured model as available within 10 seconds, THEN THE System SHALL record the failure reason in the operator log and SHALL reject each newly submitted Job with an unavailable-model message until a subsequent connectivity check confirms the model is available.
5. IF the configured Ollama_Endpoint URL is present but is not a syntactically valid HTTP or HTTPS URL, THEN THE System SHALL terminate startup without entering a ready state and SHALL emit an error message identifying the invalid configuration value.

### Requirement 3: Generate Code with the Local Ollama Model

**User Story:** As a developer, I want the agent to use my local Ollama model to generate code from my idea, so that no code or prompt leaves my infrastructure.

#### Acceptance Criteria

1. WHEN a Job transitions to Running AND the Job's working copy is non-empty, THE Coding_Agent SHALL send the complete idea text together with a selection of the existing project files from the Job's working copy as context to the Ollama_Client for the first code-generation attempt.
2. WHERE the Job's working copy is empty, WHEN a Job transitions to Running, THE Coding_Agent SHALL send the complete idea text without existing-project-file context to the Ollama_Client for the first code-generation attempt.
3. THE Coding_Agent SHALL instruct the Ollama model to return a Generation_Result in the agreed structured format, where each file entry contains a relative file path and the full intended content of that file.
4. THE Coding_Agent SHALL limit the existing project files included as context for a generation request so that they do not exceed the configured Context_Budget maximum file count and configured Context_Budget maximum total size in bytes.
5. WHERE the existing project content of the Job's working copy exceeds the configured Context_Budget, THE Coding_Agent SHALL include only a subset of the existing project files that fits within the Context_Budget and SHALL proceed with that subset as context.
6. THE Ollama_Client SHALL send the idea text and any generation request content only to the configured Ollama_Endpoint and SHALL NOT transmit idea text or generated content to any other destination.
7. WHEN the Ollama_Client returns a successful completion for a Job, THE Coding_Agent SHALL parse the completion and SHALL interpret a successfully parsed completion as a Generation_Result composed of file entries, each comprising a relative file path and the full content of that file, and SHALL proceed to the file-writing step.
8. IF the Ollama_Client returns a successful completion that cannot be parsed into a valid Generation_Result because its structure is malformed or invalid AND the number of completed attempts for the Job is less than the configured Maximum_Build_Attempts, THEN THE Coding_Agent SHALL send the parse error back to the Ollama_Client and SHALL begin a new attempt.
9. IF the Ollama_Client returns a successful completion that cannot be parsed into a valid Generation_Result because its structure is malformed or invalid AND the number of completed attempts for the Job equals the configured Maximum_Build_Attempts, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record an invalid-output reason.
10. IF the Ollama_Client receives no response within the configured generation timeout (a value between 30 and 600 seconds, default 120 seconds), THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record a timeout reason.
11. IF the Ollama_Client returns an error response, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record the returned error reason.
12. IF the Ollama_Client returns a completion that parses into a Generation_Result containing zero file entries, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record an empty-generation reason.

### Requirement 4: Iteratively Build Code Within the Container Working Directory

**User Story:** As a developer, I want the agent to write generated code, build it, and automatically retry using build errors as feedback, so that I get working code without manual steps and without risking my host system.

#### Acceptance Criteria

1. WHEN code generation for a Job attempt completes and produces a Generation_Result, THE Coding_Agent SHALL write the full content of each file entry to the file entry's relative path within the Job's working copy of the Target_Project_Directory inside the Execution_Sandbox, creating the file if absent and overwriting it if present.
2. BEFORE writing any file entry of a Generation_Result, THE Coding_Agent SHALL validate the file entry's resolved path against the Job's Execution_Sandbox boundary, and IF a file entry's path resolves outside the Job's Execution_Sandbox working directory, THEN THE Execution_Sandbox SHALL deny the write of that file entry and SHALL leave all files outside that working directory unmodified.
3. WHERE a Build_Command is configured for the Job's Registered_Target, WHEN the generated files for an attempt are written, THE Coding_Agent SHALL run that Registered_Target's configured Build_Command within the Job's Execution_Sandbox working directory.
4. WHEN the configured Build_Command exits with a status of zero, THE Coding_Agent SHALL proceed to the commit step.
5. IF the configured Build_Command exits with a non-zero status AND the number of completed attempts for the Job is less than the configured Maximum_Build_Attempts, THEN THE Coding_Agent SHALL send the captured build error output to the Ollama_Client and SHALL begin a new attempt that writes revised generated files.
6. IF the configured Build_Command exits with a non-zero status AND the number of completed attempts for the Job equals the configured Maximum_Build_Attempts, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record a max-attempts-exhausted reason that includes the build command output up to a configured maximum capture size of 1 megabyte.
7. WHERE a Build_Command is not configured for the Job's Registered_Target, THE Coding_Agent SHALL skip the build step and SHALL proceed to the commit step.
8. IF a build or test command runs longer than the configured execution timeout (a value between 1 and 3600 seconds, default 300 seconds), THEN THE Coding_Agent SHALL terminate the command, SHALL set the Job_Status to Failed, and SHALL record a timeout reason.
9. IF the configured Maximum_Build_Attempts is absent, is not an integer, or is not between 1 and 10 inclusive, THEN THE System SHALL terminate startup without entering a ready state and SHALL emit an error message identifying the invalid maximum-build-attempts value.
10. IF the Execution_Sandbox working directory cannot be initialized for a Job, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record a sandbox-initialization-failure reason.
11. IF writing a file entry of the Generation_Result into the Job's Execution_Sandbox working directory fails, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record a write-failure reason.
12. WHEN a Job transitions to Running, THE Coding_Agent SHALL update the Job's working copy to the latest state of the Job's Registered_Target's configured default branch obtained from that Target_Repository's configured remote before generating code for the Job.
13. IF updating the Job's working copy from the Target_Repository's remote fails, THEN THE Coding_Agent SHALL set the Job_Status to Failed and SHALL record a sync-failure reason.
14. IF the configured Context_Budget maximum file count is absent, is not an integer, or is not between 1 and 1000 inclusive, OR the configured Context_Budget maximum total size in bytes is absent, is not an integer, or is not between 1024 and 67108864 inclusive, THEN THE System SHALL terminate startup without entering a ready state and SHALL emit an error message identifying the invalid Context_Budget value.

### Requirement 5: Commit and Push Changes to Git

**User Story:** As a developer, I want successful changes pushed to my Git repository on a dedicated branch, so that I can review them through my normal Git workflow.

#### Acceptance Criteria

1. WHEN a Job completes the build phase successfully or the build step is skipped, THE Git_Manager SHALL create a new branch, from the up-to-date default-branch state the Job's working copy was synced to in accordance with Requirement 4.12, in the Job's Registered_Target's Target_Repository whose name is derived from the Job identifier in accordance with that Registered_Target's branch-naming scheme; because the Job identifier contains the associated target name, the resulting branch name also includes the target name.
2. WHEN file changes exist in the Job's Execution_Sandbox working copy, THE Git_Manager SHALL stage all added, modified, and deleted files and SHALL create a commit whose message references the Job identifier.
3. WHEN a commit is created, THE Git_Manager SHALL push the new branch to the configured remote of the Job's Registered_Target's Target_Repository.
4. WHEN pushing to the remote of the Job's Registered_Target's Target_Repository, THE Git_Manager SHALL authenticate using that Registered_Target's configured Git credentials.
5. IF the push to the remote fails or does not complete within the configured push timeout, THEN THE Git_Manager SHALL set the Job_Status to Failed and SHALL record the push error reason.
6. IF a Job produces no file changes, THEN THE Git_Manager SHALL skip the commit and push steps and SHALL set the Job_Status to Succeeded with a no-changes note.
7. IF the Git credentials configured for the Job's Registered_Target are not present, THEN THE Git_Manager SHALL set the Job_Status to Failed, SHALL record a missing-credentials reason, and SHALL NOT attempt the push.
8. WHEN the new branch is successfully pushed to the remote, THE Git_Manager SHALL set the Job_Status to Succeeded.
9. IF a Job's Job_Status is Cancelled, THEN THE Git_Manager SHALL NOT stage, commit, or push any changes for that Job.

### Requirement 6: Report Job Status to Discord

**User Story:** As a developer, I want to receive status updates in Discord, so that I know when my idea has been built or has failed.

#### Acceptance Criteria

1. WHEN a Job transitions to Running, THE Discord_Bot SHALL post, within 5 seconds of the transition, a started message referencing the Job identifier to the originating Discord channel.
2. WHEN a Job transitions to Succeeded, THE Discord_Bot SHALL post, within 5 seconds of the transition, a completion message to the originating Discord channel containing the Job identifier and either the pushed branch name or a no-changes note.
3. WHEN a Job transitions to Failed, THE Discord_Bot SHALL post, within 5 seconds of the transition, a failure message containing the Job identifier and the recorded failure reason to the originating Discord channel.
4. WHEN an Authorized_User invokes the `/status` slash command with a Job identifier argument, THE Discord_Bot SHALL reply, within 5 seconds of the invocation, in the originating Discord channel with the current Job_Status of that Job.
5. IF a `/status` request references a Job identifier that does not exist, THEN THE Discord_Bot SHALL reply with a not-found message identifying the requested Job identifier.
6. IF a user whose Discord identifier is not present in the configured allowlist invokes the `/status` slash command, THEN THE Discord_Bot SHALL reply with an authorization-denied message and SHALL NOT disclose the Job_Status.
7. IF posting a status message to the originating Discord channel fails, THEN THE Discord_Bot SHALL retry delivery up to 3 times and SHALL record a delivery-failure reason in the operator log if all retries are exhausted.

### Requirement 7: Manage Concurrent Jobs

**User Story:** As an operator, I want the System to manage multiple submitted ideas in order, so that concurrent submissions do not corrupt each other's work.

#### Acceptance Criteria

1. THE System SHALL assign each Job a Job identifier composed of the associated Registered_Target's name plus a unique component, such that the Job identifier contains the associated target name and is unique among all Jobs created during the System's lifetime and is not reused for any subsequent Job.
2. THE Coding_Agent SHALL execute each Job in its own isolated working copy derived from the Job's Registered_Target's directory (or its associated Target_Repository) inside the Container, separate from every other Job's working copy including other Jobs against the same Registered_Target, so that concurrent Jobs do not modify the same files and do not corrupt each other's work.
3. WHILE the number of Running Jobs is greater than or equal to the configured concurrency limit, THE System SHALL keep each newly submitted Job in Job_Status Queued.
4. WHEN a running slot becomes available and at least one Job is in Job_Status Queued, THE System SHALL transition the Queued Job with the earliest submission time to Running.
5. WHEN a Running Job transitions to Job_Status Succeeded, Failed, or Cancelled, THE System SHALL make a running slot available.
6. IF the configured concurrency limit is absent or is not an integer between 1 and 100 inclusive, THEN THE System SHALL terminate startup and SHALL emit an error message identifying the invalid concurrency-limit value.
7. THE System SHALL permit multiple concurrent Running Jobs associated with the same Registered_Target, each executing in its own isolated working copy and on its own branch, subject only to the global concurrency limit and with no per-target serialization.

### Requirement 8: Cancel Jobs

**User Story:** As a developer, I want to cancel a Job I submitted, so that I can stop work I no longer need without affecting other Jobs.

#### Acceptance Criteria

1. IF a user whose Discord identifier is not present in the configured allowlist invokes the `/cancel` slash command, THEN THE Discord_Bot SHALL NOT cancel any Job and SHALL reply with a message indicating the user is not authorized to cancel Jobs.
2. IF an Authorized_User invokes the `/cancel` slash command with a Job identifier that does not exist, THEN THE Discord_Bot SHALL reply with a not-found message identifying the requested Job identifier.
3. WHEN an Authorized_User invokes the `/cancel` slash command targeting a Job whose Job_Status is Queued, THE System SHALL remove the Job from the queue, SHALL set the Job_Status to Cancelled, and THE Discord_Bot SHALL reply in the originating Discord channel with a cancellation confirmation referencing the Job identifier.
4. WHEN an Authorized_User invokes the `/cancel` slash command targeting a Job whose Job_Status is Running, THE System SHALL signal the Job to stop, SHALL terminate any running build or test command for that Job, SHALL set the Job_Status to Cancelled, and THE Discord_Bot SHALL reply in the originating Discord channel with a cancellation confirmation referencing the Job identifier.
5. WHEN a Job is set to Cancelled, THE System SHALL NOT commit or push any changes produced by that Job.
6. IF an Authorized_User invokes the `/cancel` slash command targeting a Job whose Job_Status is Succeeded, Failed, or Cancelled, THEN THE Discord_Bot SHALL reply with a message indicating the Job cannot be cancelled because it is in a terminal state.

### Requirement 9: List Target Directories

**User Story:** As a developer, I want to see which targets are available, so that I can choose the correct target when submitting a build.

#### Acceptance Criteria

1. WHEN an Authorized_User invokes the `/targets` slash command, THE Discord_Bot SHALL reply in the originating Discord channel with the names of all currently Registered_Targets.
2. WHERE no Registered_Targets exist, WHEN an Authorized_User invokes the `/targets` slash command, THE Discord_Bot SHALL reply in the originating Discord channel with a message indicating that no targets are configured.
3. IF a user whose Discord identifier is not present in the configured allowlist invokes the `/targets` slash command, THEN THE Discord_Bot SHALL NOT disclose any target names and SHALL reply with a message indicating the user is not authorized.

### Requirement 10: Register a New Target Directory

**User Story:** As an admin, I want to register new target directories from Discord, so that the team can build against additional projects without redeploying the agent.

#### Acceptance Criteria

1. WHEN an Admin_User invokes the `/addtarget` slash command with a target name that is not already a Registered_Target name and a directory path that resolves to an existing directory inside the Workspace_Directory, THE System SHALL add the target to the Target_Registry, SHALL persist the target so it survives Container restarts, and THE Discord_Bot SHALL reply in the originating Discord channel with a confirmation referencing the target name.
2. IF a user whose Discord identifier is not present in the configured admin allowlist invokes the `/addtarget` slash command, THEN THE System SHALL NOT register any target and THE Discord_Bot SHALL reply with a message indicating the user is not authorized to register targets.
3. IF the supplied directory path resolves to a location outside the Workspace_Directory, including through parent-directory traversal or symbolic links, THEN THE System SHALL NOT register the target and THE Discord_Bot SHALL reply with a message indicating the path must be within the workspace.
4. IF the supplied directory path does not refer to an existing directory, THEN THE System SHALL NOT register the target and THE Discord_Bot SHALL reply with a message indicating the directory does not exist.
5. IF the supplied target name matches an existing Registered_Target name, THEN THE System SHALL NOT overwrite the existing target and THE Discord_Bot SHALL reply with a message indicating the name is already in use.
6. IF the supplied target name contains no non-whitespace characters or its length measured in Unicode characters exceeds 64, THEN THE System SHALL NOT register the target and THE Discord_Bot SHALL reply with a message indicating the target name is invalid.

### Requirement 11: Revise a Completed Job

**User Story:** As a developer, I want to revise a successfully completed Job with feedback, so that I can refine the produced code on the same branch without starting over.

#### Acceptance Criteria

1. WHEN an Authorized_User invokes the `/revise` slash command with a Job identifier argument referencing a Job whose Job_Status is Succeeded, whose branch information is still retained by the System, and a feedback argument containing at least one non-whitespace character, THE System SHALL create a revision that continues on the same branch the referenced Job created.
2. WHEN a revision is created, THE Coding_Agent SHALL load the contents of the referenced Job's existing branch together with the supplied feedback as context for the Ollama_Client.
3. WHEN a revision runs, THE Coding_Agent SHALL perform the same iterative build loop defined in Requirement 4, performing generate, write, build, and fix attempts up to the configured Maximum_Build_Attempts.
4. WHEN a revision completes its build phase successfully, THE Git_Manager SHALL add new commits to the referenced Job's existing branch and SHALL push those commits to the configured remote of the Target_Repository.
5. IF a user whose Discord identifier is not present in the configured allowlist invokes the `/revise` slash command, THEN THE System SHALL NOT create a revision and THE Discord_Bot SHALL reply with a message indicating the user is not authorized to revise Jobs.
6. IF a `/revise` request references a Job identifier that does not exist, THEN THE System SHALL NOT create a revision and THE Discord_Bot SHALL reply with a not-found message identifying the requested Job identifier.
7. IF a `/revise` request references a Job whose Job_Status is Queued, Running, Failed, or Cancelled, THEN THE System SHALL NOT create a revision and THE Discord_Bot SHALL reply with a message indicating that only successfully completed Jobs can be revised.
8. IF the feedback argument supplied to the `/revise` slash command contains no non-whitespace characters, THEN THE System SHALL NOT create a revision and THE Discord_Bot SHALL reply with a message requesting non-empty feedback.
9. IF the referenced Job's branch information is no longer retained by the System, THEN THE System SHALL NOT create a revision and THE Discord_Bot SHALL reply with a message indicating the Job can no longer be revised.
10. WHILE a revision is in progress, THE System SHALL apply the same concurrency limits and isolated-working-copy rules defined in Requirement 7 that apply to a normal Job.
11. WHEN a revision transitions to Running, Succeeded, or Failed, THE Discord_Bot SHALL post the corresponding started, completion, or failure message defined in Requirement 6 to the originating Discord channel.

### Requirement 12: Validate the Target Registry at Startup

**User Story:** As an operator, I want invalid targets in the Target_Registry to be detected and excluded when the System loads, so that Jobs only run against usable targets and per-target misconfiguration does not cause failures partway through a Job.

#### Acceptance Criteria

1. WHEN the System starts, THE System SHALL load the persisted Target_Registry and SHALL evaluate each Registered_Target for its required per-target configuration, including its directory path and its Target_Repository Git remote.
2. IF a Registered_Target loaded from the Target_Registry is missing required per-target configuration such that Jobs against that Registered_Target cannot run, THEN THE System SHALL exclude that Registered_Target from the usable Registered_Targets and SHALL record an error in the operator log identifying the excluded target name and the missing configuration value.
3. WHEN a Registered_Target loaded from the Target_Registry has all required per-target configuration present, THE System SHALL include that Registered_Target in the usable Registered_Targets.
4. IF an Authorized_User invokes the `/build` slash command with a target name that is not among the usable Registered_Targets, including a Registered_Target excluded by startup validation, THEN THE Discord_Bot SHALL reject the command as not-registered in accordance with Requirement 1.6.

### Requirement 13: Restrict Commands to Allowed Channels

**User Story:** As an operator, I want the bot to accept commands only in approved channels, so that command usage is confined to channels I control and is not triggered from arbitrary places in the server.

#### Acceptance Criteria

1. IF a slash command among `/build`, `/status`, `/cancel`, `/targets`, `/addtarget`, and `/revise` is invoked in a Discord channel whose identifier is not among the configured Allowed_Channels, THEN THE Discord_Bot SHALL NOT act on the command and SHALL reply in the originating channel, within 5 seconds of invocation, with a message indicating the command is not permitted in this channel.
2. WHEN a slash command among `/build`, `/status`, `/cancel`, `/targets`, `/addtarget`, and `/revise` is invoked in a Discord channel whose identifier is among the configured Allowed_Channels by a user authorized for that command, THE Discord_Bot SHALL process the command in accordance with the requirement governing that command.
3. THE Discord_Bot SHALL evaluate the Allowed_Channels restriction in addition to the existing user-allowlist and admin-allowlist authorization checks, such that a command is acted upon only when the originating channel is among the Allowed_Channels and the invoking user is authorized for that command.
4. WHEN the System starts, THE System SHALL read the configured Allowed_Channels from configuration.
5. IF no Allowed_Channels are configured at startup, THEN THE System SHALL terminate startup without entering a ready state and SHALL emit an error message indicating that at least one Allowed_Channel must be configured.
