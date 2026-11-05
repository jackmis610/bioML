"""Feature builder: contract, grouping, causality, and engineered-column semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bioml.eval.features import (
    FEATURE_COLS,
    FEATURE_GROUPS,
    KEY_COLS,
    build_feature_frame,
    build_features,
    build_training_data,
    feature_cols_excluding,
    group_of,
)
from bioml.eval.loaders import load_dataset
from bioml.generator.run import generate_dataset


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bioml_features_ds")
    generate_dataset(n_subjects=6, out_dir=out, seed=77, log_every=0)
    return out


def _toy_frames() -> tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame]:
    samples = pl.DataFrame(
        {
            "session_id": ["s"] * 6,
            "subject_id": ["u"] * 6,
            "t_s": [0, 1, 2, 3, 4, 5],
            "hr_wearable_bpm": [100.0, None, None, 103.0, None, 110.0],
            "power_wearable_w": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "cadence_rpm": [85.0] * 6,
            "cgm_glucose_mg_dl": [None] * 6,
            "is_warmup": [True] * 6,
            "is_recovery": [False] * 6,
        }
    ).with_columns(pl.col("cgm_glucose_mg_dl").cast(pl.Float64))
    subjects = pl.DataFrame(
        {
            "subject_id": ["u"],
            "age_years": [40],
            "sex": ["F"],
            "height_cm": [165.0],
            "weight_kg": [60.0],
            "ffm_kg": [45.0],
            "training_status": ["well_trained"],
            "vo2max_estimated_wearable_mlkgmin": [50.0],
            "hr_rest_bpm": [50.0],
        }
    )
    sessions = pl.DataFrame({"session_id": ["s"], "ambient_temp_c": [21.0]})
    return samples.lazy(), subjects, sessions


def test_feature_cols_are_unique_and_fully_grouped() -> None:
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    for col in FEATURE_COLS:
        assert group_of(col) in FEATURE_GROUPS


def test_feature_cols_excluding_removes_whole_groups() -> None:
    cols = feature_cols_excluding(["power", "cgm"])
    assert not set(cols) & set(FEATURE_GROUPS["power"])
    assert not set(cols) & set(FEATURE_GROUPS["cgm"])
    assert set(cols) | set(FEATURE_GROUPS["power"]) | set(FEATURE_GROUPS["cgm"]) == set(
        FEATURE_COLS
    )
    with pytest.raises(KeyError):
        feature_cols_excluding(["not_a_group"])


def test_forward_fill_and_gap_semantics() -> None:
    frame = build_feature_frame(*_toy_frames())
    assert frame["hr_ffill_bpm"].to_list() == [100.0, 100.0, 100.0, 103.0, 103.0, 110.0]
    assert frame["hr_gap_s"].to_list() == [0.0, 1.0, 2.0, 0.0, 1.0, 0.0]


def test_pct_hrr_est_uses_tanaka_hrmax_and_resting_hr() -> None:
    frame = build_feature_frame(*_toy_frames())
    hrmax_est = 208 - 0.7 * 40  # 180
    expected = (100.0 - 50.0) / (hrmax_est - 50.0) * 100.0
    assert frame["pct_hrr_est"][0] == pytest.approx(expected)


def test_lag_features_are_null_at_session_start() -> None:
    frame = build_feature_frame(*_toy_frames())
    assert frame["hr_slope_30s"].is_null().all()  # fewer than 30 rows
    assert frame["cgm_delta_5min"].is_null().all()
    assert frame["hr_roll30_std"][0] is None  # single sample has no std


def test_rolling_features_are_causal(dataset_path: Path) -> None:
    """Perturbing the future must not change features in the past."""
    ds = load_dataset(dataset_path)
    sid = ds.subjects["subject_id"][0]
    base = build_feature_frame(ds.samples, ds.subjects, ds.sessions, subject_ids=[sid])
    cut = 400
    perturbed_samples = ds.samples.with_columns(
        pl.when(pl.col("t_s") >= cut)
        .then(pl.lit(200.0))
        .otherwise(pl.col("hr_wearable_bpm"))
        .alias("hr_wearable_bpm"),
        pl.when(pl.col("t_s") >= cut)
        .then(pl.lit(999.0))
        .otherwise(pl.col("power_wearable_w"))
        .alias("power_wearable_w"),
    )
    pert = build_feature_frame(perturbed_samples, ds.subjects, ds.sessions, subject_ids=[sid])
    past_base = base.filter(pl.col("t_s") < cut).select(*FEATURE_COLS)
    past_pert = pert.filter(pl.col("t_s") < cut).select(*FEATURE_COLS)
    assert past_base.equals(past_pert)
    # and the future did actually change
    assert (
        not base.filter(pl.col("t_s") >= cut)
        .select("hr_roll30_mean")
        .equals(pert.filter(pl.col("t_s") >= cut).select("hr_roll30_mean"))
    )


def test_features_and_training_data_share_row_order(dataset_path: Path) -> None:
    ds = load_dataset(dataset_path)
    keys, feats = build_features(ds.samples, ds.subjects, ds.sessions)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    assert keys.equals(td.keys)
    assert feats.equals(td.features)
    assert list(keys.columns) == list(KEY_COLS)
    # sorted by subject, session, t_s
    assert td.keys.equals(td.keys.sort(["subject_id", "session_id", "t_s"]))


def test_feature_matrix_is_all_float64(dataset_path: Path) -> None:
    ds = load_dataset(dataset_path)
    _, feats = build_features(ds.samples, ds.subjects, ds.sessions)
    assert all(dt == pl.Float64 for dt in feats.dtypes)
    assert feats.to_numpy().dtype == np.float64
