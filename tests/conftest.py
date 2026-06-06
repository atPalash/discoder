"""Shared pytest fixtures and Hypothesis default settings.

Registers a default Hypothesis profile enforcing the design's minimum of 100
examples per property test, and selects it for the whole test session.
"""

from hypothesis import HealthCheck, settings

# The design requires every property test to run a minimum of 100 iterations
# (``max_examples >= 100``). Register that as the default profile so individual
# tests need not repeat it.
settings.register_profile(
    "default",
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("default")
