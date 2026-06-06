"""Property-based tests for startup configuration validation (task 3.2).

Covers Property 8: ``ConfigLoader.load`` (which constructs a ``SystemConfig``)
succeeds with respect to the bounded startup values **if and only if** every
bound holds:

- ``max_build_attempts`` is an integer in [1, 10]            (Req 4.9)
- ``concurrency_limit`` is an integer in [1, 100]            (Req 7.6)
- context-budget ``max_file_count`` is an integer in [1, 1000]   (Req 4.14)
- context-budget ``max_total_bytes`` is an integer in [1024, 67108864] (Req 4.14)
- ``allowed_channels`` is non-empty                          (Req 13.5)

Otherwise a :class:`StartupError` is raised whose message identifies the
offending value. The test only concerns configuration validation; it exercises
the real loader against on-disk YAML and uses no external services.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.config_loader import ConfigLoader
from discord_ollama_agent.errors import StartupError

# Inclusive bounds under test.
_MAX_BUILD_ATTEMPTS = (1, 10)
_CONCURRENCY_LIMIT = (1, 100)
_MAX_FILE_COUNT = (1, 1000)
_MAX_TOTAL_BYTES = (1024, 67_108_864)


def _bounded_int(low: int, high: int) -> st.SearchStrategy:
    """Integers straddling a [low, high] bound.

    Mixes boundary-clustered values with uniform in-range, below-range, and
    above-range integers so each drawn example is likely to land on either side
    of the bound, exercising both directions of the iff.
    """
    boundaries = st.sampled_from(
        [low - 2, low - 1, low, low + 1, high - 1, high, high + 1, 0]
    )
    return st.one_of(
        boundaries,
        st.integers(min_value=low, max_value=high),
        st.integers(max_value=low - 1),
        st.integers(min_value=high + 1),
    )


def _in_bounds(value: int, bounds: tuple[int, int]) -> bool:
    low, high = bounds
    return low <= value <= high


def _build_config(
    max_build_attempts: int,
    concurrency_limit: int,
    max_file_count: int,
    max_total_bytes: int,
    allowed_channels: list[int],
) -> dict:
    """Assemble a system-config mapping with all *other* values valid.

    Only the five fields under test vary; every remaining required value is held
    valid so that any raised StartupError is attributable solely to the field(s)
    deliberately driven out of bounds.
    """
    return {
        "ollama": {
            "endpoint_url": "http://localhost:11434",
            "model": "llama3",
        },
        "workspace_dir": "/tmp/agent-workspace",
        "max_build_attempts": max_build_attempts,
        "concurrency_limit": concurrency_limit,
        "context_budget": {
            "max_file_count": max_file_count,
            "max_total_bytes": max_total_bytes,
        },
        "allowed_channels": allowed_channels,
    }


# Feature: discord-ollama-coding-agent, Property 8
# Property 8: Startup configuration validation accepts exactly in-bounds values.
# Validates: Requirements 4.9, 4.14, 7.6, 13.5
@settings(max_examples=20)
@given(
    max_build_attempts=_bounded_int(*_MAX_BUILD_ATTEMPTS),
    concurrency_limit=_bounded_int(*_CONCURRENCY_LIMIT),
    max_file_count=_bounded_int(*_MAX_FILE_COUNT),
    max_total_bytes=_bounded_int(*_MAX_TOTAL_BYTES),
    allowed_channels=st.lists(st.integers(), max_size=5),
)
def test_startup_validation_accepts_exactly_in_bounds_values(
    max_build_attempts: int,
    concurrency_limit: int,
    max_file_count: int,
    max_total_bytes: int,
    allowed_channels: list[int],
):
    """Startup succeeds iff all bounded values are in range; otherwise a
    StartupError identifying each offending value is raised."""
    # Which bounds (if any) are violated, mapped to the substring the
    # StartupError must contain to identify that offending value.
    violations: list[str] = []
    if not _in_bounds(max_build_attempts, _MAX_BUILD_ATTEMPTS):
        violations.append("max_build_attempts")
    if not _in_bounds(concurrency_limit, _CONCURRENCY_LIMIT):
        violations.append("concurrency_limit")
    if not _in_bounds(max_file_count, _MAX_FILE_COUNT):
        violations.append("max_file_count")
    if not _in_bounds(max_total_bytes, _MAX_TOTAL_BYTES):
        violations.append("max_total_bytes")
    if not allowed_channels:
        violations.append("allowed_channels")

    expected_valid = not violations

    config = _build_config(
        max_build_attempts,
        concurrency_limit,
        max_file_count,
        max_total_bytes,
        allowed_channels,
    )

    loader = ConfigLoader()
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "system.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        if expected_valid:
            result = loader.load(config_path)
            # Startup reached a usable config carrying the in-bounds values.
            assert result.max_build_attempts == max_build_attempts
            assert result.concurrency_limit == concurrency_limit
            assert result.context_budget.max_file_count == max_file_count
            assert result.context_budget.max_total_bytes == max_total_bytes
            assert result.allowed_channels == allowed_channels
        else:
            try:
                loader.load(config_path)
            except StartupError as exc:
                message = str(exc)
                # The error must identify every offending value.
                for offender in violations:
                    assert offender in message, (
                        f"StartupError did not identify offending value "
                        f"{offender!r}: {message}"
                    )
            else:
                raise AssertionError(
                    "Expected StartupError for out-of-bounds config "
                    f"(violations={violations}), but load succeeded"
                )
