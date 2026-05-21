"""Tests for the wearable sensor overlay."""

from __future__ import annotations

import numpy as np
import pytest

from bioml.config import load_spec
from bioml.generator.physiology import run_session
from bioml.generator.sensors import (
    apply_sensor_overlay,
    simulate_cadence,
    simulate_cgm,
    simulate_power_meter,
    simulate_ppg_hr,
)
from bioml.generator.sessions import sample_session
from bioml.generator.subjects import sample_subject


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec()


@pytest.fixture(scope="module")
def e2e_run(spec: dict) -> dict:
    rng = np.random.default_rng(2026)
    for _ in range(50):
        s = sample_subject(spec, rng)
        if s.training_status == "well_trained":
            break
    session, power = sample_session(s, rng)
    truth = run_session(s, session, power, spec, rng)
    return {"subject": s, "session": session, "truth": truth, "rng": rng}


# ---------- Power meter ----------


def test_power_meter_none_device_returns_nan() -> None:
    rng = np.random.default_rng(0)
    out = simulate_power_meter(np.array([100.0, 200.0]), None, rng)
    assert np.isnan(out).all()


def test_power_meter_preserves_scale_within_tolerance() -> None:
    rng = np.random.default_rng(0)
    truth = np.full(1000, 200.0)
    out = simulate_power_meter(truth, "Favero Assioma Duo", rng)
    mean_ratio = float(np.mean(out) / 200.0)
    assert 0.97 < mean_ratio < 1.03


def test_power_meter_tacx_is_tightest() -> None:
    rng = np.random.default_rng(7)
    truth = np.full(2000, 250.0)
    tacx = simulate_power_meter(truth, "Tacx Neo 2T", rng)
    stages = simulate_power_meter(truth, "Stages Power LR", rng)
    assert float(np.std(tacx)) < float(np.std(stages))


# ---------- Cadence ----------


def test_cadence_is_in_physiological_range() -> None:
    rng = np.random.default_rng(0)
    power = np.linspace(50, 350, 1000)
    cad = simulate_cadence(power, rng)
    assert (cad >= 50).all() and (cad <= 130).all()
    assert float(np.mean(cad[power > 200])) > float(np.mean(cad[power < 100]))


# ---------- PPG HR ----------


def test_ppg_hr_none_device_returns_nan() -> None:
    rng = np.random.default_rng(0)
    out = simulate_ppg_hr(
        np.array([100.0, 110.0]), np.array([40.0, 50.0]), np.array([85.0, 90.0]), None, rng
    )
    assert np.isnan(out).all()


def test_ppg_hr_dropout_rate_higher_at_high_intensity() -> None:
    rng = np.random.default_rng(42)
    n_steady = 5000
    hr = np.full(n_steady, 130.0)
    pct = np.full(n_steady, 50.0)
    cad = np.full(n_steady, 85.0)
    out_steady = simulate_ppg_hr(hr, pct, cad, "Apple Watch S9", rng)

    pct_high = np.full(n_steady, 90.0)
    hr_high = np.full(n_steady, 175.0)
    cad_high = np.full(n_steady, 95.0)
    out_high = simulate_ppg_hr(hr_high, pct_high, cad_high, "Apple Watch S9", rng)

    dropout_steady = float(np.isnan(out_steady).mean())
    dropout_high = float(np.isnan(out_high).mean())
    assert dropout_high > dropout_steady


def test_ppg_hr_upper_arm_has_lower_dropout_than_wrist() -> None:
    rng = np.random.default_rng(11)
    n = 8000
    hr = np.full(n, 145.0)
    pct = np.full(n, 70.0)
    cad = np.full(n, 90.0)
    wrist = simulate_ppg_hr(hr, pct, cad, "Apple Watch S9", rng)
    arm = simulate_ppg_hr(hr, pct, cad, "Polar OH1", rng)
    assert float(np.isnan(arm).mean()) < float(np.isnan(wrist).mean())


def test_ppg_hr_observed_correlates_strongly_with_truth() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    hr_truth = np.linspace(60, 180, n)
    pct = np.linspace(0, 100, n)
    cad = np.full(n, 85.0)
    obs = simulate_ppg_hr(hr_truth, pct, cad, "Garmin Fenix 7", rng)
    mask = ~np.isnan(obs)
    r = float(np.corrcoef(hr_truth[mask], obs[mask])[0, 1])
    assert r > 0.95


# ---------- CGM ----------


def test_cgm_none_device_returns_nan() -> None:
    rng = np.random.default_rng(0)
    out = simulate_cgm("fasted", 600, np.full(600, 40.0), None, rng)
    assert np.isnan(out).all()


def test_cgm_baseline_differs_by_diet_state() -> None:
    rng = np.random.default_rng(0)
    fasted = simulate_cgm("fasted", 1800, np.full(1800, 20.0), "Dexcom G7", rng)
    fed = simulate_cgm("fed_carb_2_3h", 1800, np.full(1800, 20.0), "Dexcom G7", rng)
    keto = simulate_cgm("keto_adapted", 1800, np.full(1800, 20.0), "Dexcom G7", rng)
    assert float(np.mean(fed)) > float(np.mean(fasted))
    assert float(np.mean(fasted)) > float(np.mean(keto))


def test_cgm_libre_samples_more_often_than_dexcom() -> None:
    rng = np.random.default_rng(0)
    libre = simulate_cgm("fasted", 1800, np.full(1800, 30.0), "FreeStyle Libre 3", rng)
    dexcom = simulate_cgm("fasted", 1800, np.full(1800, 30.0), "Dexcom G7", rng)
    libre_unique = len(np.unique(np.round(libre, 3)))
    dexcom_unique = len(np.unique(np.round(dexcom, 3)))
    assert libre_unique > dexcom_unique


# ---------- Orchestrator end-to-end ----------


def test_apply_sensor_overlay_returns_all_columns(e2e_run: dict) -> None:
    obs = apply_sensor_overlay(e2e_run["truth"], e2e_run["session"], e2e_run["rng"])
    expected_keys = {
        "hr_wearable_bpm",
        "power_wearable_w",
        "cadence_rpm",
        "cgm_glucose_mg_dl",
        "accelerometer_movement_index",
    }
    assert set(obs.keys()) == expected_keys
    n = len(e2e_run["truth"]["t_s"])
    for key, arr in obs.items():
        assert len(arr) == n, f"{key} length mismatch"


def test_apply_sensor_overlay_respects_none_devices(e2e_run: dict) -> None:
    session_no_devices = e2e_run["session"].model_copy(
        update={
            "device_model_wearable": None,
            "device_model_power": None,
            "device_model_cgm": None,
        }
    )
    obs = apply_sensor_overlay(e2e_run["truth"], session_no_devices, e2e_run["rng"])
    assert np.isnan(obs["hr_wearable_bpm"]).all()
    assert np.isnan(obs["power_wearable_w"]).all()
    assert np.isnan(obs["cgm_glucose_mg_dl"]).all()


def test_apply_sensor_overlay_hr_observed_tracks_truth(e2e_run: dict) -> None:
    obs = apply_sensor_overlay(e2e_run["truth"], e2e_run["session"], e2e_run["rng"])
    if e2e_run["session"].device_model_wearable is None:
        pytest.skip("no wearable device for this session")
    hr_truth = e2e_run["truth"]["hr_lab_bpm"]
    hr_obs = obs["hr_wearable_bpm"]
    mask = ~np.isnan(hr_obs)
    if mask.sum() < 50:
        pytest.skip("not enough non-dropout samples")
    r = float(np.corrcoef(hr_truth[mask], hr_obs[mask])[0, 1])
    assert r > 0.85
