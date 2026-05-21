"""Tests for the spec.yaml loader."""

from __future__ import annotations

import pytest

from bioml.config import load_spec, stratum_value


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec()


def test_spec_loads_top_level_keys(spec: dict) -> None:
    expected = {
        "schema_version",
        "time_resolution_s",
        "modality",
        "subject",
        "rer_curve",
        "glycogen",
        "rer_drift",
        "output_strategy",
        "frayn",
        "session",
        "sensors",
    }
    assert expected <= set(spec.keys())


def test_training_status_weights_sum_to_one(spec: dict) -> None:
    weights = spec["subject"]["training_status"]["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_threshold_correlation_in_range(spec: dict) -> None:
    rho = spec["subject"]["thresholds_pct_vo2max"]["lt1_lt2_correlation"]
    assert -1.0 <= rho <= 1.0


def test_stratum_value_sex_split(spec: dict) -> None:
    vo2_cfg = spec["subject"]["vo2max_mlkgmin"]
    male_un = stratum_value(vo2_cfg, "untrained", "M")
    female_un = stratum_value(vo2_cfg, "untrained", "F")
    assert male_un[0] > female_un[0]


def test_stratum_value_falls_back_to_all(spec: dict) -> None:
    hr_rest_cfg = spec["subject"]["hr_rest_bpm"]
    val = stratum_value(hr_rest_cfg, "recreational", "M")
    assert val[0] == 60
