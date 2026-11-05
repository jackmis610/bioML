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


def test_save_load_roundtrip(dataset_path: Path, tmp_path: Path) -> None:
    ds = load_dataset(dataset_path)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    model = fit_multitask(td, n_estimators=20)
    save_dir = tmp_path / "model_v1"
    model.save(save_dir)
    assert (save_dir / "rer.txt").exists()
    assert (save_dir / "vo2.txt").exists()
    assert (save_dir / "lt2.txt").exists()
    assert (save_dir / "metadata.json").exists()

    reloaded = TrainedModel.load(save_dir)
    preds_orig = model.predict(td.features)
    preds_new = reloaded.predict(td.features)
    np.testing.assert_allclose(preds_orig["rer"], preds_new["rer"], rtol=1e-9)
    np.testing.assert_allclose(preds_orig["vo2_mlmin"], preds_new["vo2_mlmin"], rtol=1e-9)
    np.testing.assert_allclose(preds_orig["lt2_prob"], preds_new["lt2_prob"], rtol=1e-9)


def test_metadata_pins_feature_contract(dataset_path: Path, tmp_path: Path) -> None:
    import json

    from bioml.eval.features import FEATURE_COLS
    from bioml.train import FeatureContractError

    ds = load_dataset(dataset_path)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    model = fit_multitask(td, n_estimators=5)
    save_dir = tmp_path / "model_meta"
    model.save(save_dir, extra_metadata={"model_version": "v-test"})

    meta = json.loads((save_dir / "metadata.json").read_text())
    assert meta["feature_names"] == list(FEATURE_COLS)
    assert meta["model_version"] == "v-test"
    assert meta["n_estimators"] == 5
    assert meta["bioml_version"]
    # boosters know their column names (no more Column_0..N)
    assert model.rer.feature_name() == list(FEATURE_COLS)

    reloaded = TrainedModel.load(save_dir)
    assert reloaded.feature_names == tuple(FEATURE_COLS)
    with pytest.raises(FeatureContractError):
        reloaded.predict(td.features.drop(td.features.columns[0]))
    with pytest.raises(FeatureContractError):
        reloaded.predict(td.features.select(list(reversed(td.features.columns))))


def test_load_without_metadata_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "rer.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        TrainedModel.load(tmp_path)
