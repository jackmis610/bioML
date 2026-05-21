"""Population-mean curve baseline.

Fits population-mean RER, VO2, and LT2-flag-probability as functions of
HR / HRmax from training data, then applies via interpolation. Strictly more
powerful than the fixed HR-zone heuristic — uses data instead of canonical
table values. The "competent sports-science benchmark" the ML model should
clearly beat.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

VO2_REST_ML_PER_KG_MIN = 3.5


def _pct_hrmax(features: pl.DataFrame) -> NDArray[np.float64]:
    age = features["age_years"].to_numpy().astype(np.float64)
    hr = features["hr_wearable_bpm"].to_numpy().astype(np.float64)
    hrmax_est = 208.0 - 0.7 * age
    pct = hr / hrmax_est * 100.0
    return np.where(np.isnan(pct), 50.0, pct)


class PopulationCurveBaseline:
    """Bin-mean fit of (RER, VO2_fraction, P(LT2)) vs %HRmax."""

    def __init__(self, n_bins: int = 20, bin_range: tuple[float, float] = (20.0, 110.0)):
        self.n_bins = n_bins
        self.bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
        self.rer_curve: NDArray[np.float64] | None = None
        self.vo2_frac_curve: NDArray[np.float64] | None = None
        self.lt2_prob_curve: NDArray[np.float64] | None = None

    def _bin_centers(self) -> NDArray[np.float64]:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def fit(self, features: pl.DataFrame, targets: pl.DataFrame) -> "PopulationCurveBaseline":
        pct = _pct_hrmax(features)
        rer = targets["rer_lab"].to_numpy().astype(np.float64)
        vo2 = targets["vo2_lab_mlmin"].to_numpy().astype(np.float64)
        weight = features["weight_kg"].to_numpy().astype(np.float64)
        vo2max_est = features["vo2max_estimated_wearable_mlkgmin"].to_numpy().astype(np.float64)
        vo2max_abs = vo2max_est * weight
        vo2_rest_abs = VO2_REST_ML_PER_KG_MIN * weight
        vo2_frac = (vo2 - vo2_rest_abs) / np.maximum(vo2max_abs - vo2_rest_abs, 1.0)
        lt2 = targets["above_lt2_flag"].to_numpy().astype(np.float64)

        idx = np.clip(
            np.searchsorted(self.bin_edges[1:-1], pct, side="right"), 0, self.n_bins - 1
        )
        rer_curve = np.full(self.n_bins, np.nan)
        vo2_curve = np.full(self.n_bins, np.nan)
        lt2_curve = np.full(self.n_bins, np.nan)
        for b in range(self.n_bins):
            mask = idx == b
            if mask.any():
                rer_curve[b] = float(np.nanmean(rer[mask]))
                vo2_curve[b] = float(np.nanmean(vo2_frac[mask]))
                lt2_curve[b] = float(np.nanmean(lt2[mask]))

        for arr in (rer_curve, vo2_curve, lt2_curve):
            mask_valid = ~np.isnan(arr)
            if mask_valid.any():
                centers = self._bin_centers()
                arr[~mask_valid] = np.interp(centers[~mask_valid], centers[mask_valid], arr[mask_valid])

        self.rer_curve = rer_curve
        self.vo2_frac_curve = vo2_curve
        self.lt2_prob_curve = lt2_curve
        return self

    def predict(self, features: pl.DataFrame) -> dict[str, NDArray[np.float64]]:
        if self.rer_curve is None:
            raise RuntimeError("PopulationCurveBaseline must be fit before predict")
        pct = _pct_hrmax(features)
        centers = self._bin_centers()
        rer_pred = np.interp(pct, centers, self.rer_curve)
        vo2_frac_pred = np.interp(pct, centers, self.vo2_frac_curve)  # type: ignore[arg-type]
        lt2_prob_pred = np.interp(pct, centers, self.lt2_prob_curve)  # type: ignore[arg-type]
        lt2_prob_pred = np.clip(lt2_prob_pred, 0.0, 1.0)

        weight = features["weight_kg"].to_numpy().astype(np.float64)
        vo2max_est = features["vo2max_estimated_wearable_mlkgmin"].to_numpy().astype(np.float64)
        vo2max_abs = vo2max_est * weight
        vo2_rest_abs = VO2_REST_ML_PER_KG_MIN * weight
        vo2_pred = vo2_rest_abs + (vo2max_abs - vo2_rest_abs) * vo2_frac_pred

        return {"rer": rer_pred, "vo2_mlmin": vo2_pred, "lt2_prob": lt2_prob_pred}
