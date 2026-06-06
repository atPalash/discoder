"""Property-based tests for the Generation models (task 2.2).

Covers the serialization round-trip of :class:`GenerationResult` through the
structured wire format (pydantic v2 JSON), which is the representation used to
communicate with the locally hosted Ollama model.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.models.generation import FileEntry, GenerationResult

# Exclude lone surrogate code points (Unicode category "Cs"): they are not
# valid in well-formed text/JSON and are outside the space of real file paths
# and contents, so constraining them keeps the generator within the valid
# input space rather than testing serializer-level edge cases.
_text = st.text(alphabet=st.characters(blacklist_categories=["Cs"]), max_size=200)

_file_entries = st.builds(FileEntry, path=_text, content=_text)

_generation_results = st.builds(
    GenerationResult,
    files=st.lists(_file_entries, max_size=10),
)


# Feature: discord-ollama-coding-agent, Property 27
# Property 27: Generation_Result round-trips through serialization.
# Validates: Requirements 3.7
@settings(max_examples=20)
@given(result=_generation_results)
def test_generation_result_round_trips_through_serialization(result: GenerationResult):
    """Serializing a GenerationResult to the wire format and parsing it back
    yields an equivalent result: same ordered file entries with identical
    paths and content."""
    wire = result.model_dump_json()
    parsed = GenerationResult.model_validate_json(wire)

    # Same number of file entries.
    assert len(parsed.files) == len(result.files)

    # Same order, with identical path and content for each entry.
    for original, restored in zip(result.files, parsed.files):
        assert restored.path == original.path
        assert restored.content == original.content

    # The full models compare equal as well.
    assert parsed == result
