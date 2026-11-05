"""Inference API: contract validation, end-to-end predictions, and train/serve parity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from bioml.eval.features import build_features, build_training_data
from bioml.eval.loaders import load_dataset
from bioml.generator.run import generate_dataset
from bioml.intake import resolve_subject
from bioml.serve.app import create_app
from bioml.serve.pipeline import frames_from_request
from bioml.train import fit_multitask

SAMPLE_COLS = ["t_s", "hr_wearable_bpm", "power_wearable_w", "cadence_rpm", "cgm_glucose_mg_dl"]


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bioml_serve_ds")
    generate_dataset(n_subjects=12, out_dir=out, seed=606, log_every=0)
    return out


@pytest.fixture(scope="module")
def model_dir(dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    ds = load_dataset(dataset_path)
    td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    model = fit_multitask(td, n_estimators=25)
    out = tmp_path_factory.mktemp("bioml_serve_model") / "model"
    model.save(out, extra_metadata={"model_version": "test-v0"})
    return out


@pytest.fixture(scope="module")
def client(model_dir: Path):
    with TestClient(create_app(model_dir)) as c:
        yield c


def _first_session(dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame]:
    ds = load_dataset(dataset_path)
    # Prefer a session that actually has a wearable HR stream.
    sessions = ds.sessions.filter(pl.col("device_model_wearable").is_not_null())
    sess = sessions.row(0, named=True)
    subj = ds.subjects.filter(pl.col("subject_id") == sess["subject_id"]).row(0, named=True)
    samples = pl.read_parquet(
        dataset_path
        / "samples"
        / f"subject_id={sess['subject_id']}"
        / f"{sess['session_id']}.parquet"
    )
    return subj, sess, samples


def _request_body(
    subj: dict[str, Any], sess: dict[str, Any], samples: pl.DataFrame
) -> dict[str, Any]:
    return {
        "subject": {
            "age_years": subj["age_years"],
            "sex": subj["sex"],
            "height_cm": subj["height_cm"],
            "weight_kg": subj["weight_kg"],
            "ffm_kg": subj["ffm_kg"],
            "hr_rest_bpm": subj["hr_rest_bpm"],
            "vo2max_estimated_wearable_mlkgmin": subj["vo2max_estimated_wearable_mlkgmin"],
            "training_status": subj["training_status"],
        },
        "session": {"ambient_temp_c": sess["ambient_temp_c"]},
        "samples": samples.select(SAMPLE_COLS).to_dicts(),
    }


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-v0"
    assert body["n_features"] > 10


def test_model_info_exposes_feature_contract(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert "feature_names" in body and body["n_features"] == len(body["feature_names"])
    assert body["model_family"] == "lightgbm_multitask"
    assert body["bioml_version"]


def test_predict_end_to_end(client: TestClient, dataset_path: Path) -> None:
    subj, sess, samples = _first_session(dataset_path)
    r = client.post("/predict", json=_request_body(subj, sess, samples))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_samples"] == samples.height
    assert body["n_grid_seconds"] == samples.height
    assert body["imputed"] == {}
    assert body["policy"] == {"rer_ceiling": 0.99, "lt2_prob_ceiling": 0.5}
    regimes = {s["regime"] for s in body["samples"]}
    assert regimes <= {"substrate_calculable", "buffering_dominated"}
    for s in body["samples"]:
        if s["regime"] == "buffering_dominated":
            assert s["fat_ox_pred_g_per_min"] is None
        else:
            assert s["fat_ox_pred_g_per_min"] is not None and s["fat_ox_pred_g_per_min"] >= 0
        assert 0.0 <= s["lt2_prob"] <= 1.0
    assert set(body["summary"]) >= {
        "pct_time_buffering_dominated",
        "peak_fat_ox_g_per_min",
        "hr_at_peak_fat_ox_bpm",
    }


def test_predict_imputes_missing_context(client: TestClient, dataset_path: Path) -> None:
    subj, sess, samples = _first_session(dataset_path)
    body = _request_body(subj, sess, samples.head(300))
    for k in ("ffm_kg", "hr_rest_bpm", "training_status"):
        del body["subject"][k]
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    assert set(r.json()["imputed"]) == {"ffm_kg", "hr_rest_bpm", "training_status"}


def test_predict_reindexes_gappy_timestamps_to_1hz(client: TestClient, dataset_path: Path) -> None:
    subj, sess, samples = _first_session(dataset_path)
    sparse = samples.filter(pl.col("t_s") % 5 == 0).head(60)  # every 5th second
    r = client.post("/predict", json=_request_body(subj, sess, sparse))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_samples"] == sparse.height
    assert body["n_grid_seconds"] == int(sparse["t_s"].max() - sparse["t_s"].min() + 1)
    assert [s["t_s"] for s in body["samples"]] == sparse["t_s"].to_list()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b["samples"].__setitem__(1, {**b["samples"][1], "t_s": b["samples"][0]["t_s"]}),
        lambda b: b["samples"][0].__setitem__("hr_wearable_bpm", 400.0),
        lambda b: b["subject"].__setitem__("sex", "X"),
        lambda b: b["subject"].__setitem__("vo2max_estimated_wearable_mlkgmin", 5.0),
        lambda b: b.__setitem__("samples", []),
    ],
    ids=["non_increasing_t_s", "hr_out_of_range", "bad_sex", "vo2max_out_of_range", "no_samples"],
)
def test_predict_rejects_invalid_requests(client: TestClient, dataset_path: Path, mutate) -> None:
    subj, sess, samples = _first_session(dataset_path)
    body = _request_body(subj, sess, samples.head(50))
    mutate(body)
    assert client.post("/predict", json=body).status_code == 422


def test_serving_features_match_training(dataset_path: Path) -> None:
    """The API must build exactly the matrix the trainer built for the same raw rows."""
    ds = load_dataset(dataset_path)
    subj, sess, samples = _first_session(dataset_path)

    _, train_feats = build_features(
        ds.samples, ds.subjects, ds.sessions, subject_ids=[subj["subject_id"]]
    )

    resolved = resolve_subject(
        age_years=subj["age_years"],
        sex=subj["sex"],
        height_cm=subj["height_cm"],
        weight_kg=subj["weight_kg"],
        vo2max_estimated_wearable_mlkgmin=subj["vo2max_estimated_wearable_mlkgmin"],
        ffm_kg=subj["ffm_kg"],
        hr_rest_bpm=subj["hr_rest_bpm"],
        training_status=subj["training_status"],
    )
    samples_lf, subjects, sessions = frames_from_request(
        resolved, sess["ambient_temp_c"], samples.select(SAMPLE_COLS)
    )
    _, serve_feats = build_features(samples_lf, subjects, sessions)

    assert serve_feats.columns == train_feats.columns
    assert serve_feats.shape == train_feats.shape
    # Equality up to floating-point evaluation order (differences are ~1e-16 ulps);
    # anything larger is real skew and fails loudly with the column named.
    for col in train_feats.columns:
        a = train_feats[col].to_numpy()
        b = serve_feats[col].to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"null pattern differs: {col}"
        mask = ~np.isnan(a)
        np.testing.assert_allclose(
            a[mask], b[mask], rtol=1e-9, atol=1e-9, err_msg=f"train/serve skew in {col}"
        )
