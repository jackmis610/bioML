"""Streamlit demo: pick a synthetic CPET, see wearable-only model predictions
overlaid on lab truth.

Launch:
    uv run streamlit run src/bioml/demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st

from bioml.eval.features import build_training_data
from bioml.eval.loaders import load_dataset
from bioml.eval.metrics import bias, mae, roc_auc
from bioml.frayn import fat_oxidation_g_per_min
from bioml.train import TrainedModel

MODEL_PATH = Path("models/v1")
DATA_PATH = Path("data/synthetic/v1")

st.set_page_config(
    page_title="bioML — wearable metabolic prediction",
    layout="wide",
)

st.title("bioML")
st.caption(
    "Predict VO2, RER, and substrate utilization from consumer wearable signals. "
    "Phase 1: synthetic-data scaffold; Aevox real-CPET integration in progress."
)


@st.cache_resource
def _load_model() -> TrainedModel:
    return TrainedModel.load(MODEL_PATH)


@st.cache_data
def _load_subjects_sessions() -> tuple[pl.DataFrame, pl.DataFrame]:
    ds = load_dataset(DATA_PATH)
    return ds.subjects, ds.sessions


if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. Train one with:\n\n"
        f"```\nuv run python -m bioml train --data {DATA_PATH} --save-model {MODEL_PATH}\n```"
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. Generate one with:\n\n"
        f"```\nuv run python -m bioml generate --n-subjects 200 --out {DATA_PATH}\n```"
    )
    st.stop()

model = _load_model()
subjects, sessions = _load_subjects_sessions()

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
        f"{row['training_status']} {row['sex']} "
        f"age {row['age_years']} VO2max {row['vo2max_mlkgmin_true']:.0f} | {sid[:8]}..."
    )


subject_id = st.sidebar.selectbox(
    "Subject",
    options=filtered["subject_id"].to_list(),
    format_func=_label,
)
subj = subjects.filter(pl.col("subject_id") == subject_id).row(0, named=True)
sess = sessions.filter(pl.col("subject_id") == subject_id).row(0, named=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Subject (lab truth)")
    st.write(f"**Training status:** `{subj['training_status']}`")
    st.write(
        f"**Sex/Age/Weight:** {subj['sex']}, "
        f"{subj['age_years']}y, {subj['weight_kg']:.1f}kg ({subj['ffm_kg']:.1f}kg FFM)"
    )
    st.write(f"**VO2max (lab):** {subj['vo2max_mlkgmin_true']:.1f} mL/kg/min")
    st.write(
        f"**Thresholds:** Fatmax {subj['fatmax_pct_vo2max_true']:.0f}%, "
        f"LT1 {subj['lt1_pct_vo2max_true']:.0f}%, LT2 {subj['lt2_pct_vo2max_true']:.0f}%"
    )
    st.write(
        f"**MFO:** {subj['mfo_g_per_min_true']:.2f} g/min at Fatmax"
    )
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

samples_path = (
    DATA_PATH / "samples" / f"subject_id={subject_id}" / f"{sess['session_id']}.parquet"
)
samples = pl.read_parquet(samples_path)

ds_full = load_dataset(DATA_PATH)
td = build_training_data(
    ds_full.samples, ds_full.subjects, ds_full.sessions, subject_ids=[subject_id]
)
preds = model.predict(td.features)

t = samples["t_s"].to_numpy()
rer_true = samples["rer_lab"].to_numpy()
vo2_true = samples["vo2_lab_mlmin"].to_numpy()
fat_true = samples["fat_ox_g_per_min"].to_numpy()
hr_truth = samples["hr_lab_bpm"].to_numpy()
hr_wearable = samples["hr_wearable_bpm"].to_numpy()
power_lab = samples["power_lab_w"].to_numpy()
lt2_true = samples["above_lt2_flag"].to_numpy()

vo2_pred = preds["vo2_mlmin"]
rer_pred = preds["rer"]
lt2_prob = preds["lt2_prob"]
vo2_pred_l = vo2_pred / 1000.0
vco2_pred_l = rer_pred * vo2_pred_l
fat_pred_raw = fat_oxidation_g_per_min(vo2_pred_l, vco2_pred_l)
fat_pred = np.where(lt2_prob < 0.5, fat_pred_raw, np.nan)

st.subheader("Session metrics (wearable-only model vs lab truth)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("RER MAE", f"{mae(rer_true, rer_pred):.4f}", f"{bias(rer_true, rer_pred):+.4f} bias")
m2.metric("VO2 MAE (mL/min)", f"{mae(vo2_true, vo2_pred):.0f}", f"{bias(vo2_true, vo2_pred):+.0f}")
m3.metric("LT2 AUC", f"{roc_auc(lt2_true, lt2_prob):.3f}")
m4.metric("HR dropout", f"{np.isnan(hr_wearable).mean() * 100:.0f}%")

st.subheader("Power-time profile (lab)")
st.line_chart(pl.DataFrame({"t_s": t, "power_lab_w": power_lab}).to_pandas().set_index("t_s"))

st.subheader("VO2 (mL/min): lab truth vs model prediction")
st.line_chart(
    pl.DataFrame(
        {"t_s": t, "vo2_truth": vo2_true, "vo2_pred": vo2_pred}
    ).to_pandas().set_index("t_s")
)

st.subheader("RER: lab truth vs model prediction")
st.line_chart(
    pl.DataFrame(
        {"t_s": t, "rer_truth": rer_true, "rer_pred": rer_pred}
    ).to_pandas().set_index("t_s")
)

st.subheader("Fat oxidation (g/min): lab truth vs model prediction")
st.caption(
    "Both are null when above-LT2 (RER ≥ 1.0 or model's LT2-flag probability ≥ 0.5) — "
    "the Frayn equations break down due to bicarbonate buffering of lactic acid."
)
st.line_chart(
    pl.DataFrame(
        {"t_s": t, "fat_ox_truth": fat_true, "fat_ox_pred_via_frayn": fat_pred}
    ).to_pandas().set_index("t_s")
)

st.subheader("Heart rate: lab chest strap (truth) vs wearable PPG vs model implicit")
st.caption(
    "The wearable PPG signal (what the model sees) has dropouts (gaps) and noise "
    "vs the lab chest strap (the true HR)."
)
st.line_chart(
    pl.DataFrame(
        {"t_s": t, "hr_lab_truth": hr_truth, "hr_wearable_observed": hr_wearable}
    ).to_pandas().set_index("t_s")
)

st.markdown("---")
st.caption(
    "This entire prediction surface uses only wearable-grade inputs (PPG HR, optional power meter, "
    "optional CGM, demographics, wearable VO2max estimate). The lab truth columns shown here are "
    "for visual comparison only and are never fed to the model."
)
