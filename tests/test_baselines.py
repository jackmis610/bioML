"""Tests for HR-zone and population-curve baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bioml.baselines.hr_zone import HRZoneBaseline
from bioml.baselines.population_curve import PopulationCurveBaseline
from bioml.eval.features import build_training_data
from bioml.eval.loaders import load_dataset
from bioml.eval.metrics import evaluate_multitask
from bioml.generator.run import generate_dataset


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bioml_baselines_ds")
    generate_dataset(n_subjects=30, out_dir=out, seed=2024, log_every=0)
    return out


@pytest.fixture(scope="module")
def training_data(dataset_path: Path):
    ds = load_dataset(dataset_path)
    return build_training_data(ds.samples, ds.subjects, ds.sessions), ds


def test_hr_zone_predictions_in_physiological_range(training_data) -> None:
    td, _ = training_data
    baseline = HRZoneBaseline().fit(td.features, td.targets)
    preds = baseline.predict(td.features)
    assert (preds["rer"] >= 0.83).all() and (preds["rer"] <= 1.05).all()
    assert (preds["vo2_mlmin"] > 0).all()
    assert ((preds["lt2_prob"] == 0) | (preds["lt2_prob"] == 1)).all()


def test_hr_zone_rer_increases_with_intensity() -> None:
    features = pl.DataFrame(
        {
            "age_years": [30, 30, 30, 30, 30],
            "hr_wearable_bpm": [100, 130, 150, 170, 185],
            "weight_kg": [75.0] * 5,
            "vo2max_estimated_wearable_mlkgmin": [50.0] * 5,
        }
    )
    preds = HRZoneBaseline().predict(features)
    # Step-function predictor: equal across same-zone inputs, never decreasing.
    assert (np.diff(preds["rer"]) >= 0).all()
    assert (np.diff(preds["vo2_mlmin"]) >= 0).all()


def test_population_curve_fits_without_error(training_data) -> None:
    td, _ = training_data
    baseline = PopulationCurveBaseline(n_bins=15).fit(td.features, td.targets)
    assert baseline.rer_curve is not None
    assert baseline.rer_curve.shape == (15,)
    assert not np.isnan(baseline.rer_curve).any()


def test_population_curve_predict_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        features = pl.DataFrame(
            {
                "age_years": [30],
                "hr_wearable_bpm": [120.0],
                "weight_kg": [75.0],
                "vo2max_estimated_wearable_mlkgmin": [50.0],
            }
        )
        PopulationCurveBaseline().predict(features)


def test_population_curve_beats_hr_zone_on_training_distribution(training_data) -> None:
    """A learned bin-mean curve should beat the fixed coaching zones on RER MAE."""
    td, _ = training_data
    hr_zone = HRZoneBaseline().fit(td.features, td.targets)
    pop = PopulationCurveBaseline().fit(td.features, td.targets)
    pz = hr_zone.predict(td.features)
    pp = pop.predict(td.features)
    rer_true = td.targets["rer_lab"].to_numpy()
    pct_true = (
        td.targets["vo2_lab_mlmin"].to_numpy()
        / (td.features["vo2max_estimated_wearable_mlkgmin"].to_numpy() * td.features["weight_kg"].to_numpy())
        * 100
    )
    hz_eval = evaluate_multitask(
        rer_true,
        pz["rer"],
        td.targets["vo2_lab_mlmin"].to_numpy(),
        pz["vo2_mlmin"],
        td.targets["above_lt2_flag"].to_numpy(),
        pz["lt2_prob"],
        pct_true,
    )
    pop_eval = evaluate_multitask(
        rer_true,
        pp["rer"],
        td.targets["vo2_lab_mlmin"].to_numpy(),
        pp["vo2_mlmin"],
        td.targets["above_lt2_flag"].to_numpy(),
        pp["lt2_prob"],
        pct_true,
    )
    assert pop_eval.rer_mae < hz_eval.rer_mae, (
        f"population curve RER MAE {pop_eval.rer_mae:.4f} did not beat HR-zone {hz_eval.rer_mae:.4f}"
    )
