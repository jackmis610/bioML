"""Substrate gating and emission (design decision D4) in one place."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from bioml.inference import (
    BUFFERING_DOMINATED,
    SUBSTRATE_CALCULABLE,
    SubstratePolicy,
    emit_substrate,
    gate_substrate,
    summarize_session,
)


def test_gate_requires_both_heads_to_agree() -> None:
    rer = np.array([0.85, 0.995, 0.85, 1.05])
    lt2 = np.array([0.1, 0.1, 0.7, 0.9])
    np.testing.assert_array_equal(gate_substrate(rer, lt2), [True, False, False, False])


def test_gate_policy_is_configurable() -> None:
    rer = np.array([0.95])
    lt2 = np.array([0.3])
    assert gate_substrate(rer, lt2, SubstratePolicy(rer_ceiling=0.99, lt2_prob_ceiling=0.5))[0]
    assert not gate_substrate(rer, lt2, SubstratePolicy(rer_ceiling=0.90, lt2_prob_ceiling=0.5))[0]
    assert not gate_substrate(rer, lt2, SubstratePolicy(rer_ceiling=0.99, lt2_prob_ceiling=0.2))[0]


def test_emit_matches_frayn_by_hand() -> None:
    # RER 0.85 at VO2 2.0 L/min -> VCO2 1.7 L/min
    out = emit_substrate(rer_pred=[0.85], vo2_pred_mlmin=[2000.0], lt2_prob=[0.1])
    fat_expected = 1.695 * 2.0 - 1.701 * 1.7
    cho_expected = 4.344 * 1.7 - 3.061 * 2.0
    assert out["fat_ox_pred_g_per_min"][0] == pytest.approx(fat_expected, abs=1e-6)
    assert out["cho_ox_pred_g_per_min"][0] == pytest.approx(cho_expected, abs=1e-6)
    assert out["energy_pred_kcal_per_min"][0] == pytest.approx(
        fat_expected * 9.75 + cho_expected * 4.18, abs=1e-6
    )
    assert out["regime"][0] == SUBSTRATE_CALCULABLE


def test_emit_nulls_outside_valid_regime() -> None:
    out = emit_substrate(
        rer_pred=[0.85, 1.02], vo2_pred_mlmin=[2000.0, 3500.0], lt2_prob=[0.1, 0.8]
    )
    assert np.isfinite(out["fat_ox_pred_g_per_min"][0])
    assert np.isnan(out["fat_ox_pred_g_per_min"][1])
    assert np.isnan(out["cho_ox_pred_g_per_min"][1])
    assert np.isnan(out["energy_pred_kcal_per_min"][1])
    assert out["regime"][1] == BUFFERING_DOMINATED
    assert out["substrate_valid"].tolist() == [True, False]


def test_emit_never_negative_fat() -> None:
    out = emit_substrate(rer_pred=[0.985], vo2_pred_mlmin=[3000.0], lt2_prob=[0.1])
    assert out["fat_ox_pred_g_per_min"][0] >= 0.0


def test_summarize_session_reports_fatmax_hr() -> None:
    preds = pl.DataFrame(
        {
            "rer_pred": [0.80, 0.85, 0.90, 1.02],
            "vo2_pred_mlmin": [1000.0, 2000.0, 3000.0, 4000.0],
            "lt2_prob": [0.0, 0.1, 0.2, 0.9],
            "substrate_valid": [True, True, True, False],
            "regime": [SUBSTRATE_CALCULABLE] * 3 + [BUFFERING_DOMINATED],
            "fat_ox_pred_g_per_min": [0.30, 0.50, 0.40, float("nan")],
            "cho_ox_pred_g_per_min": [0.5, 1.2, 2.0, float("nan")],
            "energy_pred_kcal_per_min": [5.0, 10.0, 12.0, float("nan")],
        }
    )
    s = summarize_session(preds, hr_bpm=[100.0, 130.0, 150.0, 175.0], power_w=[80, 160, 240, 320])
    assert s["n_samples"] == 4
    assert s["pct_time_buffering_dominated"] == pytest.approx(25.0)
    assert s["peak_fat_ox_g_per_min"] == pytest.approx(0.50)
    assert s["hr_at_peak_fat_ox_bpm"] == 130.0
    assert s["power_at_peak_fat_ox_w"] == 160.0
    assert s["mean_fat_ox_valid_g_per_min"] == pytest.approx(0.40)
