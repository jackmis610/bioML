"""Frayn stoichiometric equations for substrate oxidation rates.

Coefficients from Frayn (1983) J Appl Physiol 55:628-634 and Jeukendrup &
Wallis (2005) Int J Sports Med 26 Suppl 1:S28-37. Protein-nitrogen term
omitted (introduces <3% error in exercise; convention).

Validity range: RER in [0.71, 1.00] AND %VO2max <= 85. Above this range,
bicarbonate buffering of lactic acid inflates VCO2 with non-metabolic CO2
and breaks the assumption that all expired CO2 is metabolic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FAT_VO2_COEF = 1.695
FAT_VCO2_COEF = -1.701
CHO_VO2_COEF_REST = -3.21
CHO_VCO2_COEF_REST = 4.55
CHO_VO2_COEF_EXERCISE = -3.061
CHO_VCO2_COEF_EXERCISE = 4.344

RER_VALID_MIN = 0.71
RER_VALID_MAX = 1.00
VO2MAX_FRAYN_CEILING_PCT = 85.0

ENERGY_KCAL_PER_G_FAT = 9.75
ENERGY_KCAL_PER_G_CHO_GLUCOSE = 4.07
ENERGY_KCAL_PER_G_CHO_GLYCOGEN = 4.18


def fat_oxidation_g_per_min(
    vo2_l_per_min: ArrayLike,
    vco2_l_per_min: ArrayLike,
) -> NDArray[np.float64]:
    """Fat oxidation rate in g/min from VO2 and VCO2 (L/min STPD).

    Clamps negative values to zero (occurs when RER > 1.0).
    """
    vo2 = np.asarray(vo2_l_per_min, dtype=np.float64)
    vco2 = np.asarray(vco2_l_per_min, dtype=np.float64)
    fat = FAT_VO2_COEF * vo2 + FAT_VCO2_COEF * vco2
    return np.maximum(fat, 0.0)


def cho_oxidation_g_per_min(
    vo2_l_per_min: ArrayLike,
    vco2_l_per_min: ArrayLike,
    regime: str = "exercise",
) -> NDArray[np.float64]:
    """Carbohydrate oxidation rate in g/min. regime is 'rest' or 'exercise'.

    Clamps negative values to zero (rare; occurs when RER < ~0.71).
    """
    vo2 = np.asarray(vo2_l_per_min, dtype=np.float64)
    vco2 = np.asarray(vco2_l_per_min, dtype=np.float64)
    if regime == "exercise":
        cho = CHO_VO2_COEF_EXERCISE * vo2 + CHO_VCO2_COEF_EXERCISE * vco2
    elif regime == "rest":
        cho = CHO_VO2_COEF_REST * vo2 + CHO_VCO2_COEF_REST * vco2
    else:
        raise ValueError(f"regime must be 'rest' or 'exercise', got {regime!r}")
    return np.maximum(cho, 0.0)


def rer(
    vo2_l_per_min: ArrayLike,
    vco2_l_per_min: ArrayLike,
) -> NDArray[np.float64]:
    """Respiratory exchange ratio = VCO2 / VO2."""
    vo2 = np.asarray(vo2_l_per_min, dtype=np.float64)
    vco2 = np.asarray(vco2_l_per_min, dtype=np.float64)
    return vco2 / vo2


def is_frayn_valid(
    rer_value: ArrayLike,
    pct_vo2max: ArrayLike,
) -> NDArray[np.bool_]:
    """True where RER and %VO2max are inside the Frayn validity envelope."""
    r = np.asarray(rer_value, dtype=np.float64)
    p = np.asarray(pct_vo2max, dtype=np.float64)
    return (r >= RER_VALID_MIN) & (r <= RER_VALID_MAX) & (p <= VO2MAX_FRAYN_CEILING_PCT)


def energy_expenditure_kcal_per_min(
    fat_g_per_min: ArrayLike,
    cho_g_per_min: ArrayLike,
    cho_source: str = "glycogen",
) -> NDArray[np.float64]:
    """Total energy expenditure in kcal/min from substrate oxidation rates."""
    fat = np.asarray(fat_g_per_min, dtype=np.float64)
    cho = np.asarray(cho_g_per_min, dtype=np.float64)
    if cho_source == "glycogen":
        cho_energy = ENERGY_KCAL_PER_G_CHO_GLYCOGEN
    elif cho_source == "glucose":
        cho_energy = ENERGY_KCAL_PER_G_CHO_GLUCOSE
    else:
        raise ValueError(f"cho_source must be 'glycogen' or 'glucose', got {cho_source!r}")
    return fat * ENERGY_KCAL_PER_G_FAT + cho * cho_energy
