# Discord-Ollama Coding Agent

A single-container Python application that turns coding ideas submitted through
Discord slash commands into committed, pushed Git branches using a locally hosted
Ollama model.

The agent runs an iterative generate → write → build → fix loop against a set of
operator-configured targets, confines all execution to a per-Job working directory
inside the container, and reports status back to the originating Discord channel.

## Development

Install the package with its development dependencies:

```bash
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```
