"""Tests for loaders, splits, feature builder, and metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bioml.eval.features import FEATURE_COLS, TARGET_COLS, build_training_data
from bioml.eval.loaders import load_dataset
from bioml.eval.metrics import (
    bias,
    brier_score,
    evaluate_multitask,
    mae,
    mae_by_intensity_bin,
    rmse,
    roc_auc,
)
from bioml.eval.splits import physiology_stratified_split, subject_kfold
from bioml.generator.run import generate_dataset


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bioml_eval_ds")
    generate_dataset(n_subjects=20, out_dir=out, seed=99, log_every=0)
    return out


# ---------- loaders ----------


def test_load_dataset_returns_three_pieces(dataset_path: Path) -> None:
    ds = load_dataset(dataset_path)
    assert ds.subjects.height == 20
    assert ds.sessions.height == 20
    n_samples = ds.samples.select(pl.len()).collect().item()
    assert n_samples > 1000


# ---------- splits ----------


def test_kfold_does_not_leak_subjects() -> None:
    ids = [f"s{i:03d}" for i in range(50)]
    splits = subject_kfold(ids, k=5, seed=0)
    assert len(splits) == 5
    for train, val in splits:
        assert set(train).isdisjoint(set(val))
        assert set(train) | set(val) == set(ids)


def test_kfold_folds_balanced() -> None:
    ids = [f"s{i:03d}" for i in range(50)]
    splits = subject_kfold(ids, k=5, seed=0)
    val_sizes = [len(v) for _, v in splits]
    assert max(val_sizes) - min(val_sizes) <= 1


def test_kfold_deterministic_with_seed() -> None:
    ids = [f"s{i:03d}" for i in range(50)]
    a = subject_kfold(ids, k=5, seed=42)
    b = subject_kfold(ids, k=5, seed=42)
    for (ta, va), (tb, vb) in zip(a, b, strict=True):
        assert ta == tb and va == vb


def test_kfold_rejects_k_too_large() -> None:
    with pytest.raises(ValueError):
        subject_kfold(["a", "b"], k=5)


def test_stratified_split_includes_all_strata_in_test() -> None:
    ids = [f"s{i:03d}" for i in range(40)]
    statuses = (
        ["untrained"] * 10 + ["recreational"] * 15 + ["well_trained"] * 10 + ["elite"] * 5
    )
    train, test = physiology_stratified_split(ids, statuses, test_fraction=0.2, seed=0)
    test_strata = {statuses[ids.index(t)] for t in test}
    assert test_strata == {"untrained", "recreational", "well_trained", "elite"}
    assert set(train).isdisjoint(set(test))


# ---------- feature builder ----------


def test_build_training_data_columns(dataset_path: Path) -> None:
    ds = load_dataset(dataset_path)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    assert set(td.features.columns) == set(FEATURE_COLS)
    assert set(td.targets.columns) == set(TARGET_COLS)
    assert td.features.height == td.targets.height
    assert td.features.height == len(td.groups)


def test_build_training_data_excludes_specified_subjects(dataset_path: Path) -> None:
    ds = load_dataset(dataset_path)
    chosen = ds.subjects["subject_id"].head(5).to_list()
    td = build_training_data(ds.samples, ds.subjects, ds.sessions, subject_ids=chosen)
    unique_subjects = set(td.groups.unique().to_list())
    assert unique_subjects == set(chosen)


def test_build_training_data_has_no_truth_leakage(dataset_path: Path) -> None:
    """Features must not include any *_true, *_lab, or diet_state columns."""
    ds = load_dataset(dataset_path)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    leaky = [c for c in td.features.columns if "_true" in c or "_lab" in c or "diet_state" in c]
    assert leaky == [], f"truth-side columns leaked into features: {leaky}"


# ---------- metrics ----------


def test_mae_rmse_bias_on_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert bias(y, y) == 0.0


def test_mae_rmse_bias_on_constant_offset() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = y + 0.5
    assert mae(y, yhat) == pytest.approx(0.5)
    assert rmse(y, yhat) == pytest.approx(0.5)
    assert bias(y, yhat) == pytest.approx(0.5)


def test_mae_by_intensity_bin_counts() -> None:
    y = np.array([0.85, 0.90, 0.95, 1.00, 1.05])
    yhat = np.array([0.86, 0.90, 0.96, 0.99, 1.06])
    pct = np.array([25, 45, 60, 75, 95])
    out = mae_by_intensity_bin(y, yhat, pct)
    n_total = sum(n for _, n, _ in out)
    assert n_total == 5


def test_roc_auc_perfect_separation() -> None:
    y = np.array([True, True, False, False])
    score = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(y, score) == pytest.approx(1.0)


def test_roc_auc_random_predictions_near_half() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.random(n) < 0.3
    score = rng.random(n)
    auc = roc_auc(y, score)
    assert 0.45 < auc < 0.55


def test_roc_auc_all_one_class_returns_nan() -> None:
    y = np.zeros(10, dtype=bool)
    score = np.random.rand(10)
    auc = roc_auc(y, score)
    assert np.isnan(auc)


def test_brier_score_perfect_calibration() -> None:
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, p) == 0.0


def test_evaluate_multitask_returns_report() -> None:
    n = 100
    rng = np.random.default_rng(1)
    rer_true = rng.uniform(0.75, 1.05, n)
    rer_pred = rer_true + rng.normal(0, 0.02, n)
    vo2_true = rng.uniform(500, 4000, n)
    vo2_pred = vo2_true + rng.normal(0, 50, n)
    lt2_true = rer_true >= 1.0
    lt2_prob = (rer_pred - 0.9) / 0.2
    lt2_prob = np.clip(lt2_prob, 0, 1)
    pct = rng.uniform(0, 100, n)
    rep = evaluate_multitask(
        rer_true, rer_pred, vo2_true, vo2_pred, lt2_true, lt2_prob, pct
    )
    assert rep.rer_mae > 0
    assert rep.vo2_mae_mlmin > 0
    assert 0.0 <= rep.lt2_auc <= 1.0
    s = rep.summary()
    assert "RER" in s and "VO2" in s and "LT2" in s
