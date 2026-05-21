"""Tests for the session generator."""

from __future__ import annotations

import numpy as np
import pytest

from bioml.config import load_spec
from bioml.generator.sessions import (
    KCAL_TO_ML_O2_AT_WATT,
    VO2_REST_ML_PER_KG_MIN,
    power_at_vo2max,
    ramp_power_profile,
    sample_session,
)
from bioml.generator.subjects import sample_subjects


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec()


@pytest.fixture(scope="module")
def subjects(spec: dict) -> list:
    return sample_subjects(spec, n=20, seed=99)


def test_power_at_vo2max_round_trips_through_steady_state_formula(subjects: list) -> None:
    for s in subjects:
        pmax = power_at_vo2max(s)
        # Round-trip: at pmax, steady-state VO2 should equal VO2max
        vo2_rest = VO2_REST_ML_PER_KG_MIN * s.weight_kg
        vo2_at_pmax = vo2_rest + 287.0 * pmax / s.gross_efficiency_pct_true
        vo2_max_abs = s.vo2max_mlkgmin_true * s.weight_kg
        assert vo2_at_pmax == pytest.approx(vo2_max_abs, rel=0.01)


def test_ramp_profile_has_three_phases(subjects: list) -> None:
    rng = np.random.default_rng(7)
    for s in subjects:
        profile = ramp_power_profile(s, rng, warmup_w=50, warmup_s=180, cooldown_s=180)
        assert profile[0] == 50.0
        assert profile[179] == 50.0
        assert profile[-1] == 50.0
        ramp_segment = profile[180:-180]
        assert (np.diff(ramp_segment) > 0).all()


def test_ramp_terminal_power_near_vo2max(subjects: list) -> None:
    rng = np.random.default_rng(7)
    for s in subjects:
        profile = ramp_power_profile(s, rng)
        ramp_seg = profile[180:-180]
        peak_power = float(ramp_seg.max())
        pmax = power_at_vo2max(s)
        assert peak_power > 0.85 * pmax, (
            f"{s.training_status}: peak {peak_power:.0f} vs pmax {pmax:.0f}"
        )
        assert peak_power < 1.15 * pmax, (
            f"{s.training_status}: peak {peak_power:.0f} vs pmax {pmax:.0f}"
        )


def test_ramp_duration_within_cpet_bounds(subjects: list) -> None:
    rng = np.random.default_rng(7)
    for s in subjects:
        profile = ramp_power_profile(s, rng)
        ramp_only = len(profile) - 360
        assert 180 <= ramp_only <= 1200, (
            f"{s.training_status} ramp {ramp_only}s out of bounds"
        )


def test_sample_session_returns_valid_session_and_aligned_power(subjects: list) -> None:
    rng = np.random.default_rng(11)
    for s in subjects[:5]:
        session, power = sample_session(s, rng)
        assert session.subject_id == s.subject_id
        assert session.protocol == "ramp_incremental"
        assert session.duration_s == len(power)
        assert session.diet_state_true is not None
        if s.habitual_diet == "keto":
            assert session.diet_state_true == "keto_adapted"


def test_sample_session_unsupported_protocol_raises(subjects: list) -> None:
    rng = np.random.default_rng(11)
    with pytest.raises(NotImplementedError):
        sample_session(subjects[0], rng, protocol="intervals")
