"""Constraint test for Fatmax <= LT1 - 2 (needed for 4-segment RER curve ordering)."""

from __future__ import annotations

import pytest

from bioml.config import load_spec
from bioml.generator.subjects import sample_subjects


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec()


def test_fatmax_strictly_below_lt1(spec: dict) -> None:
    cohort = sample_subjects(spec, n=2000, seed=29)
    for s in cohort:
        assert s.fatmax_pct_vo2max_true <= s.lt1_pct_vo2max_true - 2.0 + 1e-6, (
            f"violation: fatmax {s.fatmax_pct_vo2max_true} vs lt1 {s.lt1_pct_vo2max_true}"
        )
