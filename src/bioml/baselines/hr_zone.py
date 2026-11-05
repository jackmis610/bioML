"""HR-zone heuristic baseline.

The classic coaching model: divide HR / HRmax into 5 zones, assign a
canonical RER and VO2 fraction to each. Wearables produce zone-coded plans
this way (Garmin, Polar). It's the "naive sports science" baseline our model
has to beat.

HRmax is taken from Tanaka 208 - 0.7*age (what a wearable computes when no
lab CPET is available).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

VO2_REST_ML_PER_KG_MIN = 3.5
LT2_HR_PCT_THRESHOLD = 87.0

ZONE_BREAKPOINTS_PCT_HRMAX = (60.0, 70.0, 80.0, 90.0, 110.0)
ZONE_RER = (0.83, 0.86, 0.92, 0.98, 1.05)
ZONE_VO2_FRACTION = (0.45, 0.62, 0.77, 0.88, 0.97)


def _zone_index(pct_hrmax: NDArray[np.float64]) -> NDArray[np.int64]:
    idx = np.searchsorted(ZONE_BREAKPOINTS_PCT_HRMAX, pct_hrmax, side="right")
    return np.clip(idx, 0, len(ZONE_BREAKPOINTS_PCT_HRMAX) - 1)


class HRZoneBaseline:
    """Stateless predictor: zone -> RER, zone -> VO2, HR threshold -> LT2 flag."""

    def fit(self, features: pl.DataFrame, targets: pl.DataFrame) -> HRZoneBaseline:
        return self

    def predict(self, features: pl.DataFrame) -> dict[str, NDArray[np.float64]]:
        age = features["age_years"].to_numpy().astype(np.float64)
        hr = features["hr_wearable_bpm"].to_numpy().astype(np.float64)
        weight = features["weight_kg"].to_numpy().astype(np.float64)
        vo2max_est = features["vo2max_estimated_wearable_mlkgmin"].to_numpy().astype(np.float64)

        hrmax_est = 208.0 - 0.7 * age
        with np.errstate(invalid="ignore"):
            pct_hrmax = np.where(np.isnan(hr), 60.0, hr / hrmax_est * 100.0)

        zones = _zone_index(pct_hrmax)
        rer = np.array(ZONE_RER)[zones]
        vo2_frac = np.array(ZONE_VO2_FRACTION)[zones]
        vo2max_abs = vo2max_est * weight
        vo2_rest_abs = VO2_REST_ML_PER_KG_MIN * weight
        vo2_pred = vo2_rest_abs + (vo2max_abs - vo2_rest_abs) * vo2_frac

        lt2_prob = (pct_hrmax >= LT2_HR_PCT_THRESHOLD).astype(np.float64)

        return {"rer": rer, "vo2_mlmin": vo2_pred, "lt2_prob": lt2_prob}
