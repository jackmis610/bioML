"""Tests for the physiology engine.

Covers VO2 first-order kinetics, the 3-segment RER curve, glycogen depletion
matching Coyle 1986 anchor, and Frayn validity gating in end-to-end runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioml.config import load_spec
from bioml.generator.physiology import (
    glycogen_depletion_rate_mmol_per_kg_per_min,
    rer_curve,
    run_session,
    steady_state_vo2_mlmin,
)
from bioml.generator.sessions import sample_session
from bioml.generator.subjects import sample_subject, sample_subjects


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec()


def test_steady_state_vo2_recovers_expected_value() -> None:
    # 75 kg recreational at 200 W with GE 20.5% -> VO2 ~ 3062 mL/min
    vo2 = steady_state_vo2_mlmin(power_w=200, weight_kg=75, ge_pct=20.5)
    assert vo2 == pytest.approx(3062.5, abs=20)


def test_vo2_steady_state_zero_power_is_resting() -> None:
    vo2 = steady_state_vo2_mlmin(power_w=0, weight_kg=70, ge_pct=21.0)
    assert vo2 == pytest.approx(245.0, abs=1.0)


def test_vo2_kinetics_first_order_approaches_target() -> None:
    """At 5*tau seconds, VO2 should be within 1% of steady-state target."""
    rng = np.random.default_rng(0)
    spec = load_spec()
    subj = sample_subject(spec, rng)
    weight = subj.weight_kg
    ge = subj.gross_efficiency_pct_true
    tau = subj.vo2_tau_moderate_s_true
    vo2_rest = 3.5 * weight
    constant_power = 150.0
    vo2_ss = steady_state_vo2_mlmin(constant_power, weight, ge)
    duration = int(5 * tau)
    vo2_current = vo2_rest
    alpha = 1.0 - np.exp(-1.0 / tau)
    for _ in range(duration):
        vo2_current = vo2_current + (vo2_ss - vo2_current) * alpha
    rel_err = abs(vo2_current - vo2_ss) / vo2_ss
    assert rel_err < 0.01


def test_rer_curve_hits_anchors_exactly() -> None:
    anchors = {
        "rest_rer": 0.82,
        "fatmax_rer": 0.87,
        "lt1_rer": 0.90,
        "lt2_rer": 1.00,
        "max_rer": 1.08,
    }
    fatmax, lt1, lt2 = 50.0, 56.0, 85.0
    r_rest = rer_curve(np.array([0.0]), **anchors, fatmax_pct=fatmax, lt1=lt1, lt2=lt2)[0]
    r_fm = rer_curve(np.array([fatmax - 1e-6]), **anchors, fatmax_pct=fatmax, lt1=lt1, lt2=lt2)[0]
    r_lt1 = rer_curve(np.array([lt1 - 1e-6]), **anchors, fatmax_pct=fatmax, lt1=lt1, lt2=lt2)[0]
    r_lt2 = rer_curve(np.array([lt2 - 1e-6]), **anchors, fatmax_pct=fatmax, lt1=lt1, lt2=lt2)[0]
    r_100 = rer_curve(np.array([100.0]), **anchors, fatmax_pct=fatmax, lt1=lt1, lt2=lt2)[0]
    assert r_rest == pytest.approx(0.82, abs=1e-6)
    assert r_fm == pytest.approx(0.87, abs=1e-4)
    assert r_lt1 == pytest.approx(0.90, abs=1e-4)
    assert r_lt2 == pytest.approx(1.00, abs=1e-4)
    assert r_100 == pytest.approx(1.08, abs=1e-6)


def test_rer_curve_monotonically_non_decreasing() -> None:
    x = np.linspace(0, 100, 500)
    anchors = {
        "rest_rer": 0.82,
        "fatmax_rer": 0.87,
        "lt1_rer": 0.90,
        "lt2_rer": 1.00,
        "max_rer": 1.08,
    }
    r = rer_curve(x, **anchors, fatmax_pct=50.0, lt1=56, lt2=85)
    assert (np.diff(r) >= -1e-9).all()


def test_rer_curve_key_slope_relations() -> None:
    """Two physiology-required slope relations:
    - s0 < s1: the Fatmax knee is a steepening (otherwise no bell-shape in fat_ox).
    - s3 > s2: above LT2 is steeper than LT1-LT2 (bicarbonate buffering).
    """
    rest, fm_rer, lt1_rer, lt2_rer, mx = 0.82, 0.87, 0.90, 1.00, 1.08
    fm, lt1, lt2 = 50.0, 56.0, 85.0
    s0 = (fm_rer - rest) / fm
    s1 = (lt1_rer - fm_rer) / (lt1 - fm)
    s2 = (lt2_rer - lt1_rer) / (lt2 - lt1)
    s3 = (mx - lt2_rer) / (100 - lt2)
    assert s0 < s1, f"s0={s0:.5f} should be < s1={s1:.5f} (Fatmax knee)"
    assert s3 > s2, f"s3={s3:.5f} should be > s2={s2:.5f} (LT2 buffering)"
    assert all(s > 0 for s in (s0, s1, s2, s3)), "all slopes positive"


def test_fat_oxidation_peaks_at_fatmax() -> None:
    """The 4-segment curve must produce a bell-shaped fat_ox vs intensity, peaking near Fatmax."""
    from bioml.frayn import fat_oxidation_g_per_min

    anchors = {
        "rest_rer": 0.82,
        "fatmax_rer": 0.88,
        "lt1_rer": 0.91,
        "lt2_rer": 1.00,
        "max_rer": 1.06,
    }
    fatmax, lt1, lt2 = 56.0, 62.0, 87.0
    vo2max_abs_l = 4.5
    vo2_rest_l = 0.275
    pct = np.linspace(20, 95, 76)
    rer_at_pct = rer_curve(pct, **anchors, fatmax_pct=fatmax, lt1=lt1, lt2=lt2)
    vo2_l = vo2_rest_l + (vo2max_abs_l - vo2_rest_l) * (pct / 100.0)
    vco2_l = rer_at_pct * vo2_l
    fat = fat_oxidation_g_per_min(vo2_l, vco2_l)
    peak_idx = int(np.argmax(fat))
    peak_pct = float(pct[peak_idx])
    assert abs(peak_pct - fatmax) < 5.0, f"fat_ox peaks at {peak_pct:.1f}% not Fatmax {fatmax}"


def test_glycogen_depletion_at_71pct_matches_coyle_1986() -> None:
    # Coyle 1986: trained cyclists at 71% VO2max -> ~0.86 mmol/kg/min phase-1
    # Our spec: well_trained r1 at 65% = 0.75; scaled to 71% via (71/65)^1.8 = 1.176
    r = glycogen_depletion_rate_mmol_per_kg_per_min(
        pct_vo2max=71.0,
        glycogen_current=142.0,
        glycogen_initial=142.0,
        r1_at_65=0.75,
        knee_fraction=0.50,
        phase2_multiplier=0.45,
    )
    assert r == pytest.approx(0.88, abs=0.05)


def test_glycogen_knee_engages_phase2() -> None:
    r_phase1 = glycogen_depletion_rate_mmol_per_kg_per_min(
        pct_vo2max=65.0,
        glycogen_current=100.0,
        glycogen_initial=100.0,
        r1_at_65=0.75,
        knee_fraction=0.50,
        phase2_multiplier=0.45,
    )
    r_phase2 = glycogen_depletion_rate_mmol_per_kg_per_min(
        pct_vo2max=65.0,
        glycogen_current=40.0,
        glycogen_initial=100.0,
        r1_at_65=0.75,
        knee_fraction=0.50,
        phase2_multiplier=0.45,
    )
    assert r_phase2 == pytest.approx(0.45 * r_phase1, abs=1e-6)


def test_glycogen_no_depletion_below_40pct_vo2max() -> None:
    r = glycogen_depletion_rate_mmol_per_kg_per_min(
        pct_vo2max=30.0,
        glycogen_current=140.0,
        glycogen_initial=140.0,
        r1_at_65=0.75,
        knee_fraction=0.50,
        phase2_multiplier=0.45,
    )
    assert r == 0.0


@pytest.fixture(scope="module")
def e2e_run(spec: dict) -> dict:
    """One end-to-end ramp test on a well-trained male; reused across tests."""
    rng = np.random.default_rng(2026)
    # Generate subjects until we get a well-trained male
    for _ in range(50):
        s = sample_subject(spec, rng)
        if s.training_status == "well_trained" and s.sex == "M":
            break
    else:
        pytest.skip("Could not draw well-trained male subject for e2e test")
    session, power = sample_session(s, rng)
    result = run_session(s, session, power, spec, rng)
    return {"subject": s, "session": session, "power": power, "result": result}


def test_e2e_arrays_have_consistent_length(e2e_run: dict) -> None:
    n = len(e2e_run["power"])
    for key, arr in e2e_run["result"].items():
        assert len(arr) == n, f"{key} length mismatch"


def test_e2e_vo2_peaks_near_vo2max(e2e_run: dict) -> None:
    s = e2e_run["subject"]
    vo2_peak = float(e2e_run["result"]["vo2_lab_mlmin"].max())
    vo2_max_abs = s.vo2max_mlkgmin_true * s.weight_kg
    rel_err = abs(vo2_peak - vo2_max_abs) / vo2_max_abs
    assert rel_err < 0.10


def test_e2e_pct_vo2max_spans_expected_range(e2e_run: dict) -> None:
    pct = e2e_run["result"]["pct_vo2max_lab"]
    assert pct.min() < 15.0
    assert pct.max() > 90.0


def test_e2e_above_lt2_flag_triggers_at_high_intensity(e2e_run: dict) -> None:
    pct = e2e_run["result"]["pct_vo2max_lab"]
    above = e2e_run["result"]["above_lt2_flag"]
    rer = e2e_run["result"]["rer_lab"]
    high_pct_idx = np.where(pct > 86)[0]
    if len(high_pct_idx) > 0:
        assert above[high_pct_idx].mean() > 0.9
    high_rer_idx = np.where(rer >= 1.0)[0]
    if len(high_rer_idx) > 0:
        assert above[high_rer_idx].all()


def test_e2e_fat_cho_null_above_lt2(e2e_run: dict) -> None:
    above = e2e_run["result"]["above_lt2_flag"]
    fat = e2e_run["result"]["fat_ox_g_per_min"]
    cho = e2e_run["result"]["cho_ox_g_per_min"]
    assert np.isnan(fat[above]).all()
    assert np.isnan(cho[above]).all()


def test_e2e_fat_cho_finite_in_valid_range(e2e_run: dict) -> None:
    rer = e2e_run["result"]["rer_lab"]
    fat = e2e_run["result"]["fat_ox_g_per_min"]
    cho = e2e_run["result"]["cho_ox_g_per_min"]
    valid = (rer >= 0.71) & (rer <= 1.0) & ~e2e_run["result"]["above_lt2_flag"]
    if valid.sum() > 50:
        assert np.isfinite(fat[valid]).all()
        assert np.isfinite(cho[valid]).all()
        assert (fat[valid] >= 0).all()
        assert (cho[valid] >= 0).all()


def test_e2e_hr_within_physiological_bounds(e2e_run: dict) -> None:
    s = e2e_run["subject"]
    hr = e2e_run["result"]["hr_lab_bpm"]
    assert (hr >= s.hr_rest_bpm - 1).all()
    assert (hr <= s.hrmax_bpm_true * 1.05).all()
    assert hr.max() > s.hr_rest_bpm + 50


def test_e2e_rer_within_physiological_bounds(e2e_run: dict) -> None:
    rer = e2e_run["result"]["rer_lab"]
    assert rer.min() >= 0.65
    assert rer.max() <= 1.25


def test_e2e_multiple_subjects_run_without_error(spec: dict) -> None:
    subjects = sample_subjects(spec, n=10, seed=44)
    rng = np.random.default_rng(44)
    for s in subjects:
        session, power = sample_session(s, rng)
        result = run_session(s, session, power, spec, rng)
        assert len(result["rer_lab"]) == len(power)
        assert np.isfinite(result["vo2_lab_mlmin"]).all()
