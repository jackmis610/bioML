"""Wearable-only feature builder shared by training, evaluation, and serving.

This module is the model's input contract. Every feature is computable from
(a) the wearable stream and (b) subject/session context a user would plausibly
supply at intake. Lab-truth columns (``*_lab``, ``*_true``) and latent state
(diet, glycogen) never enter here; ``test_build_training_data_has_no_truth_leakage``
enforces that.

Two properties matter for deployment and are tested:

* **Causality.** Rolling / lag features see only the current second and earlier
  seconds of the same session. Nothing peeks forward in time.
* **Train/serve parity.** The API builds its feature matrix through the same
  ``build_features`` call the trainer uses, so skew is a tested property rather
  than a hope (see ``tests/test_serve.py::test_serving_features_match_training``).

Features are organised in named groups so ablations can drop a whole sensor at
once ("what if the rider has no power meter?").
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain

import polars as pl

TRAINING_STATUS_ORDER = {
    "untrained": 0,
    "recreational": 1,
    "well_trained": 2,
    "elite": 3,
}

# Tanaka 2001: what a wearable uses for HRmax when no lab test exists.
HRMAX_TANAKA_INTERCEPT = 208.0
HRMAX_TANAKA_SLOPE = 0.7

HR_ROLL_SHORT_S = 30
HR_ROLL_LONG_S = 60
POWER_ROLL_S = 30
CGM_DELTA_S = 300
MIN_POWER_FOR_RATIO_W = 20.0

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # Intake-form context. ffm may be estimated from sex + weight at serving time.
    "demographics": (
        "age_years",
        "sex_M",
        "height_cm",
        "weight_kg",
        "ffm_kg",
        "training_status_ordinal",
    ),
    # Wearable-reported fitness context: the watch's VO2max estimate and resting HR.
    "fitness_context": ("vo2max_estimated_wearable_mlkgmin", "hr_rest_bpm"),
    # Instantaneous PPG HR plus explicit missingness and heart-rate-reserve framing.
    "hr": ("hr_wearable_bpm", "hr_ffill_bpm", "hr_gap_s", "pct_hrr_est"),
    # Causal rolling statistics: HR kinetics carry information about VO2 kinetics.
    "hr_dynamics": ("hr_roll30_mean", "hr_roll30_std", "hr_roll60_mean", "hr_slope_30s"),
    # Power meter / cadence sensor (absent for ~20% of sessions).
    "power": (
        "power_wearable_w",
        "power_roll30_mean",
        "power_delta_30s",
        "cadence_rpm",
        "hr_per_watt",
    ),
    # CGM, last-observation-carried-forward (absent for ~65% of sessions).
    "cgm": ("cgm_glucose_mg_dl", "cgm_delta_5min"),
    # Session clock. Informative on a ramp CPET; a protocol shortcut in general.
    "time": ("t_s", "is_warmup", "is_recovery"),
    "environment": ("ambient_temp_c",),
}

FEATURE_COLS: tuple[str, ...] = tuple(chain.from_iterable(FEATURE_GROUPS.values()))
TARGET_COLS = ("rer_lab", "vo2_lab_mlmin", "above_lt2_flag")
# Truth-side columns carried alongside for evaluation only; never model inputs.
EVAL_COLS = ("pct_vo2max_lab", "fat_ox_g_per_min", "cho_ox_g_per_min")
KEY_COLS = ("subject_id", "session_id", "t_s")

# The minimum raw columns the builder needs from each table (serving sends exactly these).
RAW_SAMPLE_COLS = (
    "session_id",
    "subject_id",
    "t_s",
    "hr_wearable_bpm",
    "power_wearable_w",
    "cadence_rpm",
    "cgm_glucose_mg_dl",
    "is_warmup",
    "is_recovery",
)
RAW_SUBJECT_COLS = (
    "subject_id",
    "age_years",
    "sex",
    "height_cm",
    "weight_kg",
    "ffm_kg",
    "training_status",
    "vo2max_estimated_wearable_mlkgmin",
    "hr_rest_bpm",
)
RAW_SESSION_COLS = ("session_id", "ambient_temp_c")


def feature_cols_excluding(groups: Iterable[str]) -> tuple[str, ...]:
    """Feature list with the named groups removed (used for ablations)."""
    drop = set(groups)
    unknown = drop - set(FEATURE_GROUPS)
    if unknown:
        raise KeyError(f"unknown feature group(s): {sorted(unknown)}")
    return tuple(chain.from_iterable(cols for g, cols in FEATURE_GROUPS.items() if g not in drop))


def group_of(feature: str) -> str:
    for g, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return g
    raise KeyError(feature)


@dataclass(frozen=True)
class TrainingData:
    """Aligned feature matrix, targets, group labels, and row keys.

    All frames share row order (sorted by subject, session, t_s). ``eval_cols``
    holds truth-side reference columns for metrics and is ``None`` at serving.
    """

    features: pl.DataFrame
    targets: pl.DataFrame
    groups: pl.Series
    keys: pl.DataFrame
    eval_cols: pl.DataFrame | None = None

    @property
    def n_rows(self) -> int:
        return self.features.height


def _subject_features(subjects: pl.DataFrame) -> pl.DataFrame:
    return subjects.select(
        "subject_id",
        pl.col("age_years").cast(pl.Float64),
        (pl.col("sex") == "M").cast(pl.Int8).alias("sex_M"),
        pl.col("height_cm").cast(pl.Float64),
        pl.col("weight_kg").cast(pl.Float64),
        pl.col("ffm_kg").cast(pl.Float64),
        pl.col("vo2max_estimated_wearable_mlkgmin").cast(pl.Float64),
        pl.col("hr_rest_bpm").cast(pl.Float64),
        pl.col("training_status")
        .replace_strict(TRAINING_STATUS_ORDER, return_dtype=pl.Int8)
        .alias("training_status_ordinal"),
    )


def _engineer(joined: pl.LazyFrame) -> pl.LazyFrame:
    """Add causal engineered columns. Input must contain raw sample + context columns."""
    by_session = "session_id"
    hr = pl.col("hr_wearable_bpm")
    hr_ffill = hr.forward_fill().over(by_session)
    last_valid_t = (
        pl.when(hr.is_not_null())
        .then(pl.col("t_s"))
        .otherwise(None)
        .forward_fill()
        .over(by_session)
    )
    hrmax_est = HRMAX_TANAKA_INTERCEPT - HRMAX_TANAKA_SLOPE * pl.col("age_years")

    step1 = joined.with_columns(
        hr_ffill.alias("hr_ffill_bpm"),
        (pl.col("t_s") - last_valid_t).cast(pl.Float64).alias("hr_gap_s"),
        pl.col("power_wearable_w")
        .rolling_mean(window_size=POWER_ROLL_S, min_samples=1)
        .over(by_session)
        .alias("power_roll30_mean"),
        (
            pl.col("cgm_glucose_mg_dl")
            - pl.col("cgm_glucose_mg_dl").shift(CGM_DELTA_S).over(by_session)
        ).alias("cgm_delta_5min"),
        pl.col("is_warmup").cast(pl.Int8),
        pl.col("is_recovery").cast(pl.Int8),
    )

    hrf = pl.col("hr_ffill_bpm")
    hr_rest = pl.col("hr_rest_bpm")
    return step1.with_columns(
        ((hrf - hr_rest) / (hrmax_est - hr_rest) * 100.0).alias("pct_hrr_est"),
        hrf.rolling_mean(window_size=HR_ROLL_SHORT_S, min_samples=1)
        .over(by_session)
        .alias("hr_roll30_mean"),
        hrf.rolling_std(window_size=HR_ROLL_SHORT_S, min_samples=2)
        .over(by_session)
        .alias("hr_roll30_std"),
        hrf.rolling_mean(window_size=HR_ROLL_LONG_S, min_samples=1)
        .over(by_session)
        .alias("hr_roll60_mean"),
        ((hrf - hrf.shift(HR_ROLL_SHORT_S).over(by_session)) / HR_ROLL_SHORT_S).alias(
            "hr_slope_30s"
        ),
        (
            pl.col("power_roll30_mean")
            - pl.col("power_roll30_mean").shift(POWER_ROLL_S).over(by_session)
        ).alias("power_delta_30s"),
        pl.when(pl.col("power_roll30_mean") > MIN_POWER_FOR_RATIO_W)
        .then(hrf / pl.col("power_roll30_mean"))
        .otherwise(None)
        .alias("hr_per_watt"),
    )


def build_feature_frame(
    samples: pl.LazyFrame,
    subjects: pl.DataFrame,
    sessions: pl.DataFrame,
    subject_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Join samples with context, engineer features, return the full sorted frame.

    The returned frame carries every input column plus the engineered ones, so
    callers can slice features, targets, and evaluation references from one
    row-aligned object.
    """
    if subject_ids is not None:
        samples = samples.filter(pl.col("subject_id").is_in(subject_ids))
        subjects = subjects.filter(pl.col("subject_id").is_in(subject_ids))

    sessions_features = sessions.select(*RAW_SESSION_COLS)
    joined = (
        samples.sort(["subject_id", "session_id", "t_s"])
        .join(_subject_features(subjects).lazy(), on="subject_id", how="left")
        .join(sessions_features.lazy(), on="session_id", how="left")
    )
    return _engineer(joined).collect()


def build_features(
    samples: pl.LazyFrame,
    subjects: pl.DataFrame,
    sessions: pl.DataFrame,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
    subject_ids: list[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Serving-path entry point: returns (keys, features) with no truth columns required."""
    frame = build_feature_frame(samples, subjects, sessions, subject_ids=subject_ids)
    return frame.select(*KEY_COLS), frame.select(*feature_cols).cast(pl.Float64)


def build_training_data(
    samples: pl.LazyFrame,
    subjects: pl.DataFrame,
    sessions: pl.DataFrame,
    subject_ids: list[str] | None = None,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
) -> TrainingData:
    """Training-path entry point: features + targets + eval references, row-aligned."""
    frame = build_feature_frame(samples, subjects, sessions, subject_ids=subject_ids)
    eval_present = [c for c in EVAL_COLS if c in frame.columns]
    return TrainingData(
        features=frame.select(*feature_cols).cast(pl.Float64),
        targets=frame.select(*TARGET_COLS),
        groups=frame["subject_id"],
        keys=frame.select(*KEY_COLS),
        eval_cols=frame.select(*eval_present) if eval_present else None,
    )
