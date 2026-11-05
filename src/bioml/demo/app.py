"""Streamlit demo: pick a synthetic CPET, see wearable-only predictions over lab truth.

Launch:
    uv run streamlit run src/bioml/demo/app.py

Everything shown as "pred" comes from the same inference path the API uses
(``bioml.inference.predict_session``), so the demo cannot drift from production.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

from bioml.eval.features import build_features
from bioml.eval.loaders import load_dataset
from bioml.eval.metrics import bias, mae, roc_auc
from bioml.inference import predict_session, summarize_session
from bioml.train import TrainedModel

MODEL_PATH = Path(os.environ.get("BIOML_MODEL_PATH", "models/v2"))
DATA_PATH = Path(os.environ.get("BIOML_DATA_PATH", "data/synthetic/v1"))

st.set_page_config(page_title="bioML — wearable metabolic prediction", layout="wide")
st.title("bioML")
st.caption(
    "Predict VO2, RER, and substrate utilization from consumer wearable signals. "
    "Phase 1: synthetic-data scaffold; Aevox real-CPET integration in progress."
)


@st.cache_resource
def _load_model(path: str) -> TrainedModel:
    return TrainedModel.load(path)


@st.cache_data
def _load_context(path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    ds = load_dataset(path)
    return ds.subjects, ds.sessions


if not (MODEL_PATH / "metadata.json").exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. Train one with:\n\n"
        f"```\nuv run python -m bioml train --data {DATA_PATH} --n-estimators 400 "
        f"--save-model {MODEL_PATH}\n```"
    )
    st.stop()
if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. Generate one with:\n\n"
        f"```\nuv run python -m bioml generate --n-subjects 100 --out {DATA_PATH}\n```"
    )
    st.stop()

model = _load_model(str(MODEL_PATH))
subjects, sessions = _load_context(str(DATA_PATH))

st.sidebar.header("Subject filter")
status_filter = st.sidebar.multiselect(
    "Training status",
    options=["untrained", "recreational", "well_trained", "elite"],
    default=["untrained", "recreational", "well_trained", "elite"],
)
filtered = subjects.filter(pl.col("training_status").is_in(status_filter))
if filtered.is_empty():
    st.warning("No subjects match the filter.")
    st.stop()


def _label(sid: str) -> str:
    row = filtered.filter(pl.col("subject_id") == sid).row(0, named=True)
    return (
        f"{row['training_status']} {row['sex']} age {row['age_years']} "
        f"VO2max {row['vo2max_mlkgmin_true']:.0f} | {sid[:8]}..."
    )


subject_id = st.sidebar.selectbox(
    "Subject", options=filtered["subject_id"].to_list(), format_func=_label
)
subj = subjects.filter(pl.col("subject_id") == subject_id).row(0, named=True)
sess = sessions.filter(pl.col("subject_id") == subject_id).row(0, named=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Model `{MODEL_PATH}` · {len(model.feature_names)} features · "
    f"version `{model.metadata.get('model_version', model.metadata.get('git_sha'))}`"
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Subject (lab truth)")
    st.write(f"**Training status:** `{subj['training_status']}`")
    st.write(
        f"**Sex/Age/Weight:** {subj['sex']}, {subj['age_years']}y, "
        f"{subj['weight_kg']:.1f}kg ({subj['ffm_kg']:.1f}kg FFM)"
    )
    st.write(f"**VO2max (lab):** {subj['vo2max_mlkgmin_true']:.1f} mL/kg/min")
    st.write(
        f"**Thresholds:** Fatmax {subj['fatmax_pct_vo2max_true']:.0f}%, "
        f"LT1 {subj['lt1_pct_vo2max_true']:.0f}%, LT2 {subj['lt2_pct_vo2max_true']:.0f}%"
    )
    st.write(f"**MFO:** {subj['mfo_g_per_min_true']:.2f} g/min at Fatmax")
    st.write(
        f"**Wearable VO2max estimate** (what the model sees): "
        f"{subj['vo2max_estimated_wearable_mlkgmin']:.1f} mL/kg/min"
    )
with col2:
    st.subheader("Session")
    st.write(f"**Protocol:** `{sess['protocol']}`")
    st.write(f"**Duration:** {sess['duration_s']}s ({sess['duration_s'] / 60:.1f} min)")
    st.write(f"**Latent diet state:** `{sess['diet_state_true']}` _(model does not see this)_")
    st.write(f"**Ambient temp:** {sess['ambient_temp_c']:.1f}°C")
    st.write("**Devices:**")
    st.write(f"- Wearable HR: `{sess['device_model_wearable']}`")
    st.write(f"- Power: `{sess['device_model_power']}`")
    st.write(f"- CGM: `{sess['device_model_cgm']}`")

samples_path = DATA_PATH / "samples" / f"subject_id={subject_id}" / f"{sess['session_id']}.parquet"
samples = pl.read_parquet(samples_path)

ds_full = load_dataset(DATA_PATH)
keys, features = build_features(
    ds_full.samples,
    ds_full.subjects,
    ds_full.sessions,
    feature_cols=model.feature_names,
    subject_ids=[subject_id],
)
preds = predict_session(model, features)
summary = summarize_session(
    preds,
    hr_bpm=features["hr_ffill_bpm"].to_numpy(),
    power_w=features["power_roll30_mean"].to_numpy(),
)

t = samples["t_s"].to_numpy()
rer_true = samples["rer_lab"].to_numpy()
vo2_true = samples["vo2_lab_mlmin"].to_numpy()
fat_true = samples["fat_ox_g_per_min"].to_numpy()
hr_truth = samples["hr_lab_bpm"].to_numpy()
hr_wearable = samples["hr_wearable_bpm"].to_numpy()
power_lab = samples["power_lab_w"].to_numpy()
lt2_true = samples["above_lt2_flag"].to_numpy()

rer_pred = preds["rer_pred"].to_numpy()
vo2_pred = preds["vo2_pred_mlmin"].to_numpy()
lt2_prob = preds["lt2_prob"].to_numpy()
fat_pred = preds["fat_ox_pred_g_per_min"].to_numpy()

truth_valid = np.isfinite(fat_true)
truth_peak_idx = (
    int(np.nanargmax(np.where(truth_valid, fat_true, -np.inf))) if truth_valid.any() else None
)

st.subheader("Session metrics (wearable-only model vs lab truth)")
st.caption(
    "In-sample: the saved model was fit on all subjects, so these per-session numbers are "
    "optimistic. Out-of-fold results are in reports/v2/REPORT.md."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("RER MAE", f"{mae(rer_true, rer_pred):.4f}", f"{bias(rer_true, rer_pred):+.4f} bias")
m2.metric("VO2 MAE (mL/min)", f"{mae(vo2_true, vo2_pred):.0f}", f"{bias(vo2_true, vo2_pred):+.0f}")
m3.metric("LT2 AUC", f"{roc_auc(lt2_true, lt2_prob):.3f}")
m4.metric("HR dropout", f"{np.isnan(hr_wearable).mean() * 100:.0f}%")

st.subheader("Fatmax estimate (the product output)")
f1, f2, f3 = st.columns(3)
f1.metric(
    "Peak fat ox (pred)",
    f"{summary['peak_fat_ox_g_per_min']:.2f} g/min" if summary["peak_fat_ox_g_per_min"] else "n/a",
    f"truth {fat_true[truth_peak_idx]:.2f}" if truth_peak_idx is not None else None,
)
f2.metric(
    "HR at peak fat ox (pred)",
    f"{summary['hr_at_peak_fat_ox_bpm']:.0f} bpm" if summary["hr_at_peak_fat_ox_bpm"] else "n/a",
    f"truth {hr_truth[truth_peak_idx]:.0f}" if truth_peak_idx is not None else None,
)
f3.metric("Time buffering-dominated", f"{summary['pct_time_buffering_dominated']:.0f}%")

st.subheader("Power-time profile (lab)")
st.line_chart(pl.DataFrame({"t_s": t, "power_lab_w": power_lab}).to_pandas().set_index("t_s"))

st.subheader("VO2 (mL/min): lab truth vs model prediction")
st.line_chart(
    pl.DataFrame({"t_s": t, "vo2_truth": vo2_true, "vo2_pred": vo2_pred})
    .to_pandas()
    .set_index("t_s")
)

st.subheader("RER: lab truth vs model prediction")
st.line_chart(
    pl.DataFrame({"t_s": t, "rer_truth": rer_true, "rer_pred": rer_pred})
    .to_pandas()
    .set_index("t_s")
)

st.subheader("Fat oxidation (g/min): lab truth vs model prediction")
st.caption(
    "Both are null above LT2. Truth: generator's Frayn envelope. Prediction: the inference gate "
    "(rer_pred < 0.99 AND P[LT2] < 0.5). Bicarbonate buffering of lactic acid breaks Frayn there."
)
st.line_chart(
    pl.DataFrame({"t_s": t, "fat_ox_truth": fat_true, "fat_ox_pred_via_frayn": fat_pred})
    .to_pandas()
    .set_index("t_s")
)

st.subheader("P(above LT2): model vs truth flag")
st.line_chart(
    pl.DataFrame({"t_s": t, "lt2_prob": lt2_prob, "lt2_truth": lt2_true.astype(float)})
    .to_pandas()
    .set_index("t_s")
)

st.subheader("Heart rate: lab chest strap (truth) vs wearable PPG")
st.caption("The PPG stream (what the model sees) has dropouts and noise vs the chest strap.")
st.line_chart(
    pl.DataFrame({"t_s": t, "hr_lab_truth": hr_truth, "hr_wearable_observed": hr_wearable})
    .to_pandas()
    .set_index("t_s")
)

st.markdown("---")
st.caption(
    "This entire prediction surface uses only wearable-grade inputs (PPG HR, optional power meter, "
    "optional CGM, demographics, wearable VO2max estimate). Lab truth is shown for comparison only "
    "and is never fed to the model."
)
