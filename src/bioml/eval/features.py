"""Build wearable-only feature matrix and multitask targets.

The model never sees lab-truth columns or latent state (diet_state, glycogen).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

TRAINING_STATUS_ORDER = {
    "untrained": 0,
    "recreational": 1,
    "well_trained": 2,
    "elite": 3,
}

FEATURE_COLS = (
    "t_s",
    "hr_wearable_bpm",
    "power_wearable_w",
    "cadence_rpm",
    "cgm_glucose_mg_dl",
    "is_warmup",
    "is_recovery",
    "age_years",
    "sex_M",
    "height_cm",
    "weight_kg",
    "ffm_kg",
    "vo2max_estimated_wearable_mlkgmin",
    "training_status_ordinal",
    "ambient_temp_c",
)

TARGET_COLS = ("rer_lab", "vo2_lab_mlmin", "above_lt2_flag")


@dataclass(frozen=True)
class TrainingData:
    features: pl.DataFrame
    targets: pl.DataFrame
    groups: pl.Series

    @property
    def n_rows(self) -> int:
        return self.features.height


def build_training_data(
    samples: pl.LazyFrame,
    subjects: pl.DataFrame,
    sessions: pl.DataFrame,
    subject_ids: list[str] | None = None,
) -> TrainingData:
    """Join samples + subjects + sessions, select wearable-only features and targets."""
    if subject_ids is not None:
        samples = samples.filter(pl.col("subject_id").is_in(subject_ids))
        subjects = subjects.filter(pl.col("subject_id").is_in(subject_ids))

    subjects_features = subjects.select(
        "subject_id",
        "age_years",
        (pl.col("sex") == "M").cast(pl.Int8).alias("sex_M"),
        "height_cm",
        "weight_kg",
        "ffm_kg",
        "vo2max_estimated_wearable_mlkgmin",
        pl.col("training_status").replace_strict(TRAINING_STATUS_ORDER).alias(
            "training_status_ordinal"
        ),
    )
    sessions_features = sessions.select("session_id", "ambient_temp_c")

    joined = (
        samples.join(subjects_features.lazy(), on="subject_id", how="left")
        .join(sessions_features.lazy(), on="session_id", how="left")
        .collect()
    )

    features = joined.select(*FEATURE_COLS)
    targets = joined.select(*TARGET_COLS)
    groups = joined["subject_id"]
    return TrainingData(features=features, targets=targets, groups=groups)
