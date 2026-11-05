"""Inference-time substrate emission: turn (RER, VO2, P[above LT2]) into fuel use.

Implements design decision D4 (research/design_decisions.md). The model predicts
the smooth physiological quantities on the whole intensity range; the Frayn
stoichiometry is applied downstream only where it is valid. Above LT2,
bicarbonate buffering of lactic acid inflates VCO2 with non-metabolic CO2, so
fat/CHO rates are reported as null with an explicit ``regime`` label rather
than a confidently wrong number.

This module is shared by the evaluation report, the Streamlit demo, and the
FastAPI service so the gating rule exists in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from bioml.frayn import (
    cho_oxidation_g_per_min,
    energy_expenditure_kcal_per_min,
    fat_oxidation_g_per_min,
)

if TYPE_CHECKING:
    from bioml.train import TrainedModel

SUBSTRATE_CALCULABLE = "substrate_calculable"
BUFFERING_DOMINATED = "buffering_dominated"

PREDICTION_COLS = (
    "rer_pred",
    "vo2_pred_mlmin",
    "lt2_prob",
    "substrate_valid",
    "regime",
    "fat_ox_pred_g_per_min",
    "cho_ox_pred_g_per_min",
    "energy_pred_kcal_per_min",
)


@dataclass(frozen=True)
class SubstratePolicy:
    """When to trust Frayn. Defaults are the spec values (output_strategy in spec.yaml)."""

    rer_ceiling: float = 0.99
    lt2_prob_ceiling: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {"rer_ceiling": self.rer_ceiling, "lt2_prob_ceiling": self.lt2_prob_ceiling}


DEFAULT_POLICY = SubstratePolicy()


def gate_substrate(
    rer_pred: ArrayLike,
    lt2_prob: ArrayLike,
    policy: SubstratePolicy = DEFAULT_POLICY,
) -> NDArray[np.bool_]:
    """True where both the RER head and the LT2 head agree Frayn is applicable."""
    r = np.asarray(rer_pred, dtype=np.float64)
    p = np.asarray(lt2_prob, dtype=np.float64)
    return (r < policy.rer_ceiling) & (p < policy.lt2_prob_ceiling)


def emit_substrate(
    rer_pred: ArrayLike,
    vo2_pred_mlmin: ArrayLike,
    lt2_prob: ArrayLike,
    policy: SubstratePolicy = DEFAULT_POLICY,
) -> dict[str, NDArray[Any]]:
    """Per-second fat/CHO oxidation and energy expenditure, null outside the valid regime."""
    r = np.asarray(rer_pred, dtype=np.float64)
    vo2_l = np.asarray(vo2_pred_mlmin, dtype=np.float64) / 1000.0
    vco2_l = r * vo2_l
    valid = gate_substrate(r, lt2_prob, policy)

    fat = fat_oxidation_g_per_min(vo2_l, vco2_l)
    cho = cho_oxidation_g_per_min(vo2_l, vco2_l, regime="exercise")
    energy = energy_expenditure_kcal_per_min(fat, cho, cho_source="glycogen")

    nan = np.nan
    regime = np.where(valid, SUBSTRATE_CALCULABLE, BUFFERING_DOMINATED)
    return {
        "substrate_valid": valid,
        "regime": regime,
        "fat_ox_pred_g_per_min": np.where(valid, fat, nan),
        "cho_ox_pred_g_per_min": np.where(valid, cho, nan),
        "energy_pred_kcal_per_min": np.where(valid, energy, nan),
    }


def predict_session(
    model: TrainedModel,
    features: pl.DataFrame,
    policy: SubstratePolicy = DEFAULT_POLICY,
) -> pl.DataFrame:
    """Run the three heads and the substrate gate. Rows align with ``features``."""
    heads = model.predict(features)
    emitted = emit_substrate(heads["rer"], heads["vo2_mlmin"], heads["lt2_prob"], policy)
    return pl.DataFrame(
        {
            "rer_pred": heads["rer"],
            "vo2_pred_mlmin": heads["vo2_mlmin"],
            "lt2_prob": heads["lt2_prob"],
            "substrate_valid": emitted["substrate_valid"],
            "regime": emitted["regime"],
            "fat_ox_pred_g_per_min": emitted["fat_ox_pred_g_per_min"],
            "cho_ox_pred_g_per_min": emitted["cho_ox_pred_g_per_min"],
            "energy_pred_kcal_per_min": emitted["energy_pred_kcal_per_min"],
        }
    )


def summarize_session(
    predictions: pl.DataFrame,
    hr_bpm: ArrayLike | None = None,
    power_w: ArrayLike | None = None,
) -> dict[str, Any]:
    """Session-level outputs a coach or app would surface.

    The most product-relevant one is the HR (and power) at peak predicted fat
    oxidation: a wearable-only estimate of the subject's Fatmax zone.
    """
    n = predictions.height
    valid = predictions["substrate_valid"].to_numpy()
    fat = predictions["fat_ox_pred_g_per_min"].to_numpy()
    vo2 = predictions["vo2_pred_mlmin"].to_numpy()
    out: dict[str, Any] = {
        "n_samples": n,
        "pct_time_buffering_dominated": float(100.0 * (1.0 - valid.mean())) if n else float("nan"),
        "mean_vo2_pred_mlmin": float(vo2.mean()) if n else float("nan"),
        "peak_vo2_pred_mlmin": float(vo2.max()) if n else float("nan"),
        "mean_fat_ox_valid_g_per_min": None,
        "peak_fat_ox_g_per_min": None,
        "hr_at_peak_fat_ox_bpm": None,
        "power_at_peak_fat_ox_w": None,
    }
    if valid.any():
        finite = np.isfinite(fat)
        out["mean_fat_ox_valid_g_per_min"] = float(np.nanmean(fat[finite]))
        peak_idx = int(np.nanargmax(np.where(finite, fat, -np.inf)))
        out["peak_fat_ox_g_per_min"] = float(fat[peak_idx])
        if hr_bpm is not None:
            hr = np.asarray(hr_bpm, dtype=np.float64)
            out["hr_at_peak_fat_ox_bpm"] = (
                float(hr[peak_idx]) if np.isfinite(hr[peak_idx]) else None
            )
        if power_w is not None:
            pw = np.asarray(power_w, dtype=np.float64)
            out["power_at_peak_fat_ox_w"] = (
                float(pw[peak_idx]) if np.isfinite(pw[peak_idx]) else None
            )
    return out
