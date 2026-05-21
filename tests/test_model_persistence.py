"""Verify saved LightGBM models load and predict identically to in-memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioml.eval.features import build_training_data
from bioml.eval.loaders import load_dataset
from bioml.generator.run import generate_dataset
from bioml.train import TrainedModel, fit_multitask


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bioml_persistence_ds")
    generate_dataset(n_subjects=10, out_dir=out, seed=314, log_every=0)
    return out


def test_save_load_roundtrip(
    dataset_path: Path, tmp_path: Path
) -> None:
    ds = load_dataset(dataset_path)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    model = fit_multitask(td, n_estimators=20)
    save_dir = tmp_path / "model_v1"
    model.save(save_dir)
    assert (save_dir / "rer.txt").exists()
    assert (save_dir / "vo2.txt").exists()
    assert (save_dir / "lt2.txt").exists()

    reloaded = TrainedModel.load(save_dir)
    preds_orig = model.predict(td.features)
    preds_new = reloaded.predict(td.features)
    np.testing.assert_allclose(preds_orig["rer"], preds_new["rer"], rtol=1e-9)
    np.testing.assert_allclose(preds_orig["vo2_mlmin"], preds_new["vo2_mlmin"], rtol=1e-9)
    np.testing.assert_allclose(preds_orig["lt2_prob"], preds_new["lt2_prob"], rtol=1e-9)
