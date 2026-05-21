"""Sanity tests for Frayn substrate-oxidation equations.

Anchor values from Frayn 1983 and Jeukendrup & Wallis 2005.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioml.frayn import (
    cho_oxidation_g_per_min,
    energy_expenditure_kcal_per_min,
    fat_oxidation_g_per_min,
    is_frayn_valid,
    rer,
)


def test_pure_fat_oxidation_rer_071() -> None:
    vo2 = np.array([1.0])
    vco2 = np.array([0.71])
    fat = fat_oxidation_g_per_min(vo2, vco2)
    cho = cho_oxidation_g_per_min(vo2, vco2, regime="exercise")
    assert fat[0] == pytest.approx(0.488, abs=0.01)
    assert cho[0] == pytest.approx(0.0, abs=0.05)


def test_pure_cho_oxidation_rer_100() -> None:
    vo2 = np.array([1.0])
    vco2 = np.array([1.0])
    fat = fat_oxidation_g_per_min(vo2, vco2)
    cho = cho_oxidation_g_per_min(vo2, vco2, regime="exercise")
    assert fat[0] == 0.0
    assert cho[0] == pytest.approx(1.283, abs=0.01)


def test_clamps_negative_fat_above_rer_one() -> None:
    vo2 = np.array([1.0])
    vco2 = np.array([1.10])
    fat = fat_oxidation_g_per_min(vo2, vco2)
    assert fat[0] == 0.0


def test_rest_vs_exercise_cho_coefficients_differ() -> None:
    vo2 = np.array([1.0])
    vco2 = np.array([0.95])
    cho_rest = cho_oxidation_g_per_min(vo2, vco2, regime="rest")
    cho_ex = cho_oxidation_g_per_min(vo2, vco2, regime="exercise")
    assert cho_rest[0] > cho_ex[0]
    assert cho_rest[0] == pytest.approx(1.112, abs=0.01)
    assert cho_ex[0] == pytest.approx(1.066, abs=0.01)


def test_invalid_regime_raises() -> None:
    with pytest.raises(ValueError, match="regime must be"):
        cho_oxidation_g_per_min(np.array([1.0]), np.array([0.9]), regime="space")


def test_validity_envelope_rer_bounds() -> None:
    rer_vals = np.array([0.65, 0.71, 0.85, 1.00, 1.05])
    pct = np.full(5, 60.0)
    valid = is_frayn_valid(rer_vals, pct)
    np.testing.assert_array_equal(valid, [False, True, True, True, False])


def test_validity_envelope_pct_vo2max_ceiling() -> None:
    rer_vals = np.full(3, 0.90)
    pct = np.array([60.0, 85.0, 86.0])
    valid = is_frayn_valid(rer_vals, pct)
    np.testing.assert_array_equal(valid, [True, True, False])


def test_rer_calculation() -> None:
    r = rer(np.array([1.0, 2.0]), np.array([0.85, 1.7]))
    np.testing.assert_allclose(r, [0.85, 0.85])


def test_energy_expenditure_from_substrates() -> None:
    fat = np.array([0.3])
    cho = np.array([1.5])
    ee = energy_expenditure_kcal_per_min(fat, cho, cho_source="glycogen")
    expected = 0.3 * 9.75 + 1.5 * 4.18
    assert ee[0] == pytest.approx(expected, abs=0.001)


def test_energy_expenditure_invalid_source() -> None:
    with pytest.raises(ValueError, match="cho_source must be"):
        energy_expenditure_kcal_per_min(np.array([0.1]), np.array([0.5]), cho_source="acetate")


def test_vectorized_inputs() -> None:
    vo2 = np.array([0.8, 1.2, 2.0, 3.5])
    vco2 = np.array([0.6, 1.0, 1.8, 3.5])
    fat = fat_oxidation_g_per_min(vo2, vco2)
    cho = cho_oxidation_g_per_min(vo2, vco2, regime="exercise")
    assert fat.shape == vo2.shape
    assert cho.shape == vo2.shape
    assert np.all(fat >= 0)
    assert np.all(cho >= 0)
