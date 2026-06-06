"""Property-based test for target-name validation (task 4.6).

Covers Property 26: target-name validation enforces length and non-whitespace.

*For any* Unicode string supplied as a :class:`RegisteredTarget` name --
including whitespace-only strings and strings exceeding 64 characters --
construction succeeds **if and only if** the name contains at least one
non-whitespace character *and* its length, measured in Unicode characters, is at
most :data:`TARGET_NAME_MAX_LENGTH` (64). Any other name is rejected with a
:class:`pydantic.ValidationError`. This is the guarantee behind Req 10.6.

All other constructor fields are fixed to known-valid values so the only thing
that can make construction fail is the name itself.

**Validates: Requirements 10.6**
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from discord_ollama_agent.models.target import (
    TARGET_NAME_MAX_LENGTH,
    RegisteredTarget,
)

# Arbitrary Unicode names, biased to exercise both validation bounds:
#   - lengths span 0 (empty) through well past TARGET_NAME_MAX_LENGTH, so the
#     >64-character rejection path is hit regularly;
#   - the alphabet mixes whitespace and non-whitespace code points (including
#     astral/multibyte characters) so whitespace-only names and mixed names are
#     both generated.
_NAME = st.text(
    alphabet=st.characters(min_codepoint=0, max_codepoint=0x2FFFF),
    min_size=0,
    max_size=TARGET_NAME_MAX_LENGTH + 16,
)

# Whitespace-only names of varying length, to reliably exercise the
# non-whitespace requirement (these are otherwise rare under the broad alphabet).
_WHITESPACE_ONLY = st.text(
    alphabet=" \t\n\r\f\v\u00a0\u2003\u3000",
    min_size=1,
    max_size=TARGET_NAME_MAX_LENGTH + 16,
)


def _construct(name: str) -> RegisteredTarget:
    """Build a target whose only variable field is ``name``."""
    return RegisteredTarget(
        name=name,
        directory_path=Path("workspace/target"),
        repo_remote="git@example.com:org/repo.git",
        credentials_ref="GIT_TOKEN",
    )


# Feature: discord-ollama-coding-agent, Property 26: Target name validation enforces length and non-whitespace
# Validates: Requirements 10.6
@settings(max_examples=20, deadline=None)
@given(name=st.one_of(_NAME, _WHITESPACE_ONLY))
def test_target_name_validation_enforces_length_and_non_whitespace(name: str):
    """Acceptance holds iff the name is non-blank and at most 64 Unicode chars.

    ``len(name)`` counts Unicode code points in CPython, which is the unit the
    model measures against :data:`TARGET_NAME_MAX_LENGTH`.
    """
    expected_valid = bool(name.strip()) and len(name) <= TARGET_NAME_MAX_LENGTH

    if expected_valid:
        target = _construct(name)
        assert target.name == name
    else:
        with pytest.raises(ValidationError):
            _construct(name)
