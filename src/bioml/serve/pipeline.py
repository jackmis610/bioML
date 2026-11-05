"""Request -> feature frame -> predictions, through the training feature builder.

The only transformation specific to serving is reshaping the request into the
three canonical tables (subjects / sessions / samples). Feature engineering is
``bioml.eval.features.build_features`` - the same call the trainer makes - so
train/serve parity is a property of the code, checked by ``tests/test_serve.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from bioml import __version__
from bioml.eval.features import RAW_SAMPLE_COLS, build_features
from bioml.inference import DEFAULT_POLICY, SubstratePolicy, predict_session, summarize_session
from bioml.intake import ResolvedSubject, resolve_subject
from bioml.serve.schemas import PredictRequest, PredictResponse, SamplePrediction
from bioml.train import TrainedModel

REQUEST_SUBJECT_ID = "request-subject"
REQUEST_SESSION_ID = "request-session"
# Generator convention (generator/run.py): first/last 180 s flagged as warmup/recovery.
WARMUP_S = 180
RECOVERY_S = 180


def samples_to_grid(samples: pl.DataFrame) -> pl.DataFrame:
    """Reindex arbitrary (strictly increasing) t_s onto a contiguous 1 Hz grid.

    Missing seconds become null sensor rows - exactly how dropouts are represented
    in training - so rolling windows keep their 1-row-per-second meaning.
    """
    t_min = int(samples["t_s"].min())  # type: ignore[arg-type]
    t_max = int(samples["t_s"].max())  # type: ignore[arg-type]
    grid = pl.DataFrame({"t_s": np.arange(t_min, t_max + 1, dtype=np.int64)})
    joined = grid.join(samples.with_columns(pl.col("t_s").cast(pl.Int64)), on="t_s", how="left")
    n = joined.height
    return joined.with_columns(
        pl.lit(REQUEST_SESSION_ID).alias("session_id"),
        pl.lit(REQUEST_SUBJECT_ID).alias("subject_id"),
        (pl.col("t_s") - t_min < WARMUP_S).alias("is_warmup"),
        (pl.col("t_s") - t_min >= max(0, n - RECOVERY_S)).alias("is_recovery"),
    ).select(*RAW_SAMPLE_COLS)


def frames_from_request(
    subject: ResolvedSubject,
    ambient_temp_c: float | None,
    samples: pl.DataFrame,
) -> tuple[pl.LazyFrame, pl.DataFrame, pl.DataFrame]:
    """The three canonical tables for one request."""
    subjects = pl.DataFrame([subject.as_row(REQUEST_SUBJECT_ID)])
    sessions = pl.DataFrame(
        {"session_id": [REQUEST_SESSION_ID], "ambient_temp_c": [ambient_temp_c]},
        schema={"session_id": pl.String, "ambient_temp_c": pl.Float64},
    )
    return samples_to_grid(samples).lazy(), subjects, sessions


def _nan_to_none(x: float) -> float | None:
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)


def predict_request(model: TrainedModel, request: PredictRequest) -> PredictResponse:
    subject = resolve_subject(**request.subject.model_dump())
    raw = pl.DataFrame(
        [s.model_dump() for s in request.samples],
        schema={
            "t_s": pl.Int64,
            "hr_wearable_bpm": pl.Float64,
            "power_wearable_w": pl.Float64,
            "cadence_rpm": pl.Float64,
            "cgm_glucose_mg_dl": pl.Float64,
        },
    )
    samples_lf, subjects, sessions = frames_from_request(
        subject, request.session.ambient_temp_c, raw
    )
    keys, features = build_features(
        samples_lf, subjects, sessions, feature_cols=model.feature_names
    )

    policy = SubstratePolicy(**request.policy.model_dump()) if request.policy else DEFAULT_POLICY
    preds = predict_session(model, features, policy)

    hr_for_summary = (
        features["hr_ffill_bpm"].to_numpy()
        if "hr_ffill_bpm" in features.columns
        else np.full(features.height, np.nan)
    )
    power_for_summary = (
        features["power_roll30_mean"].to_numpy()
        if "power_roll30_mean" in features.columns
        else np.full(features.height, np.nan)
    )
    summary = summarize_session(preds, hr_bpm=hr_for_summary, power_w=power_for_summary)
    summary = {k: (_nan_to_none(v) if isinstance(v, float) else v) for k, v in summary.items()}

    # Return one row per *requested* second (drop grid fill-ins).
    requested = set(raw["t_s"].to_list())
    out = pl.concat([keys.select("t_s"), preds], how="horizontal").filter(
        pl.col("t_s").is_in(list(requested))
    )

    rows: list[SamplePrediction] = []
    for r in out.iter_rows(named=True):
        rows.append(
            SamplePrediction(
                t_s=int(r["t_s"]),
                rer_pred=float(r["rer_pred"]),
                vo2_pred_mlmin=float(r["vo2_pred_mlmin"]),
                lt2_prob=float(r["lt2_prob"]),
                regime=str(r["regime"]),
                fat_ox_pred_g_per_min=_nan_to_none(r["fat_ox_pred_g_per_min"]),
                cho_ox_pred_g_per_min=_nan_to_none(r["cho_ox_pred_g_per_min"]),
                energy_pred_kcal_per_min=_nan_to_none(r["energy_pred_kcal_per_min"]),
            )
        )

    meta: dict[str, Any] = model.metadata
    return PredictResponse(
        model_version=str(meta.get("model_version", meta.get("git_sha", __version__))),
        n_samples=len(rows),
        n_grid_seconds=features.height,
        imputed=subject.imputed,
        policy=policy.to_dict(),
        summary=summary,
        samples=rows,
    )
