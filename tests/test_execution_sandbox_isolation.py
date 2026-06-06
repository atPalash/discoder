"""Property-based test for per-Job working-copy isolation (task 5.5).

Covers Property 5: working copies are isolated per Job.

*For any* set of concurrently tracked Jobs -- including multiple Jobs against the
same Registered_Target (same ``target_name``, and even a shared Job-id prefix) --
the working-copy root that :meth:`ExecutionSandbox.init_working_dir` assigns to
each Job is distinct from every other Job's working-copy root. This is what lets
concurrent Jobs operate without modifying each other's files or corrupting each
other's work (Req 1.9, 7.2, 7.7).

The oracle is independent of the production code: the test simply collects the
root path returned for every Job (created under one shared parent ``root_dir``,
as the System would) and asserts the collection is pairwise distinct. The
generator deliberately forces same-target and same-id collisions so the
distinctness guarantee is exercised, not merely the easy all-unique case.

**Validates: Requirements 1.9, 7.2, 7.7**
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from discord_ollama_agent.execution_sandbox import ExecutionSandbox
from discord_ollama_agent.models.job import Job

# A small pool of target names so independently drawn Jobs frequently collide on
# the same Registered_Target, exercising the "multiple Jobs against the same
# target" clause of Property 5.
_TARGET_NAMES = st.sampled_from(["svc", "api", "web-app", "svc.backend"])

# Id suffixes drawn from a tiny alphabet (and allowed to be empty) so two Jobs
# can end up with an identical full id -- the strongest stress for the
# distinctness guarantee, which must not rely on Job-id uniqueness.
_ID_SUFFIX = st.text(alphabet="ab012", min_size=0, max_size=3)

# Each spec is one tracked Job: a target name, an id suffix, and whether its
# source target directory exists/has content (a missing source is the valid
# "new code" case, which must still yield a distinct empty working copy).
_JOB_SPEC = st.fixed_dictionaries(
    {
        "target_name": _TARGET_NAMES,
        "id_suffix": _ID_SUFFIX,
        "has_source": st.booleans(),
    }
)


def _make_job(target_name: str, id_suffix: str) -> Job:
    """Build a Job whose id embeds its target name (Req 7.1 id shape)."""
    return Job(
        id=f"{target_name}-{id_suffix}",
        target_name=target_name,
        idea="do the thing",
        submitted_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        channel_id=1,
        user_id=2,
    )


# Feature: discord-ollama-coding-agent, Property 5: Working copies are isolated per Job
# Validates: Requirements 1.9, 7.2, 7.7
@settings(max_examples=20, deadline=None)
@given(specs=st.lists(_JOB_SPEC, min_size=2, max_size=6))
def test_working_copy_roots_are_distinct_per_job(specs: list[dict]) -> None:
    """Every tracked Job gets a working-copy root distinct from all others.

    Creates one sandbox per Job under a single shared parent ``root_dir`` (the
    per-Job working-copy area inside the Container), initializes each Job's
    working copy from its target's source directory, and asserts the resulting
    roots are pairwise distinct -- even when Jobs share a target name or a
    complete Job id (Req 1.9, 7.2, 7.7).
    """
    with tempfile.TemporaryDirectory() as base_name:
        base = Path(base_name)
        # Shared parent under which all per-Job working-copy roots are created.
        root_dir = base / "work"

        # One source directory per distinct target, shared by all Jobs against
        # that target -- exactly the "same Registered_Target" scenario.
        sources: dict[str, Path] = {}
        for index, spec in enumerate(specs):
            name = spec["target_name"]
            if name not in sources:
                src = base / "targets" / f"target_{index}"
                src.mkdir(parents=True, exist_ok=True)
                (src / "main.py").write_text("print('hi')\n", encoding="utf-8")
                (src / "README.md").write_text(f"# {name}\n", encoding="utf-8")
                sources[name] = src

        roots: list[Path] = []
        for spec in specs:
            job = _make_job(spec["target_name"], spec["id_suffix"])
            source = sources[spec["target_name"]] if spec["has_source"] else (
                base / "does-not-exist"
            )
            sandbox = ExecutionSandbox(root_dir)
            working = sandbox.init_working_dir(job, source)

            # The bound working_dir is exactly the returned root.
            assert sandbox.working_dir == working
            # Roots live under the shared parent (Req 7.2).
            assert working.parent == root_dir
            roots.append(working)

        # The core property: pairwise-distinct working-copy roots across all
        # tracked Jobs, including same-target and same-id collisions.
        assert len(roots) == len(specs)
        assert len(set(roots)) == len(roots)
        # Distinct as resolved canonical paths too (no aliasing via symlinks).
        resolved = [r.resolve() for r in roots]
        assert len(set(resolved)) == len(resolved)
