"""Fill the context fields a real user will not reliably supply.

Customers rarely arrive with a clean intake form. Rather than reject a request
for a missing fat-free mass or training status, we impute from the same
population priors the generator was built on (research/spec.yaml), and report
what was imputed so the caller can see it. Keeping the priors in one file means
the generator and the service cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bioml.config import load_spec, stratum_value
from bioml.schemas import Sex, TrainingStatus

STRATA: tuple[TrainingStatus, ...] = ("untrained", "recreational", "well_trained", "elite")

# Mirrors the generator's body-composition assumption (generator/subjects.py).
BODY_FAT_PCT_MEAN = {"M": 18.0, "F": 26.0}
BODY_FAT_TRAINED_OFFSET_PCT = -5.0
MIN_FFM_KG = 25.5

_SPEC: dict[str, Any] | None = None


def _spec() -> dict[str, Any]:
    global _SPEC  # noqa: PLW0603 - module-level cache of a static YAML file
    if _SPEC is None:
        _SPEC = load_spec()
    return _SPEC


def infer_training_status(vo2max_est_mlkgmin: float, sex: Sex, age_years: int) -> TrainingStatus:
    """Nearest-stratum assignment from the wearable VO2max estimate.

    The spec's VO2max priors are stated at age 25 with a 0.45 mL/kg/min/yr
    decline, so the estimate is first projected back to age 25 before comparing
    against stratum means.
    """
    vo2_cfg = _spec()["subject"]["vo2max_mlkgmin"]
    dec_mean = vo2_cfg["age_decrement_per_year"][0]
    vo2_at_25 = vo2max_est_mlkgmin + dec_mean * max(0, age_years - 25)
    best: TrainingStatus = "recreational"
    best_dist = float("inf")
    for stratum in STRATA:
        mean, _sd = stratum_value(vo2_cfg, stratum, sex)
        dist = abs(vo2_at_25 - mean)
        if dist < best_dist:
            best, best_dist = stratum, dist
    return best


def estimate_ffm_kg(weight_kg: float, sex: Sex, training_status: TrainingStatus) -> float:
    bf = BODY_FAT_PCT_MEAN[sex]
    if training_status in ("well_trained", "elite"):
        bf += BODY_FAT_TRAINED_OFFSET_PCT
    return max(MIN_FFM_KG, round(weight_kg * (1.0 - bf / 100.0), 1))


def default_hr_rest_bpm(sex: Sex, training_status: TrainingStatus) -> float:
    mean, _sd = stratum_value(_spec()["subject"]["hr_rest_bpm"], training_status, sex)
    return float(mean)


@dataclass(frozen=True)
class ResolvedSubject:
    """Subject context with every model-required field present, plus an imputation log."""

    age_years: int
    sex: Sex
    height_cm: float
    weight_kg: float
    ffm_kg: float
    hr_rest_bpm: float
    vo2max_estimated_wearable_mlkgmin: float
    training_status: TrainingStatus
    imputed: dict[str, str] = field(default_factory=dict)

    def as_row(self, subject_id: str) -> dict[str, Any]:
        return {
            "subject_id": subject_id,
            "age_years": self.age_years,
            "sex": self.sex,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "ffm_kg": self.ffm_kg,
            "training_status": self.training_status,
            "vo2max_estimated_wearable_mlkgmin": self.vo2max_estimated_wearable_mlkgmin,
            "hr_rest_bpm": self.hr_rest_bpm,
        }


def resolve_subject(
    *,
    age_years: int,
    sex: Sex,
    height_cm: float,
    weight_kg: float,
    vo2max_estimated_wearable_mlkgmin: float,
    ffm_kg: float | None = None,
    hr_rest_bpm: float | None = None,
    training_status: TrainingStatus | None = None,
) -> ResolvedSubject:
    imputed: dict[str, str] = {}
    if training_status is None:
        training_status = infer_training_status(vo2max_estimated_wearable_mlkgmin, sex, age_years)
        imputed["training_status"] = (
            f"nearest stratum to wearable VO2max estimate (age-adjusted) -> {training_status}"
        )
    if ffm_kg is None:
        ffm_kg = estimate_ffm_kg(weight_kg, sex, training_status)
        imputed["ffm_kg"] = f"weight x (1 - population body-fat prior) -> {ffm_kg}"
    if hr_rest_bpm is None:
        hr_rest_bpm = default_hr_rest_bpm(sex, training_status)
        imputed["hr_rest_bpm"] = f"stratum prior mean -> {hr_rest_bpm}"
    return ResolvedSubject(
        age_years=age_years,
        sex=sex,
        height_cm=height_cm,
        weight_kg=weight_kg,
        ffm_kg=ffm_kg,
        hr_rest_bpm=hr_rest_bpm,
        vo2max_estimated_wearable_mlkgmin=vo2max_estimated_wearable_mlkgmin,
        training_status=training_status,
        imputed=imputed,
    )
