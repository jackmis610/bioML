"""Request / response contracts for the inference API.

Bounds mirror ``bioml.schemas`` so a request is rejected (HTTP 422) for the same
physiologically impossible values the data layer rejects.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from bioml.schemas import Sex, TrainingStatus

MAX_SAMPLES = 6 * 3600  # six hours at 1 Hz


class SubjectContext(BaseModel):
    age_years: int = Field(ge=18, le=99)
    sex: Sex
    height_cm: float = Field(gt=120, lt=230)
    weight_kg: float = Field(gt=35, lt=200)
    vo2max_estimated_wearable_mlkgmin: float = Field(
        ge=15, le=100, description="The watch's VO2max estimate, not a lab value."
    )
    ffm_kg: float | None = Field(default=None, gt=25, lt=150, description="Imputed if omitted.")
    hr_rest_bpm: float | None = Field(
        default=None, ge=28, le=110, description="Imputed if omitted."
    )
    training_status: TrainingStatus | None = Field(
        default=None, description="Inferred from the VO2max estimate if omitted."
    )


class SessionContext(BaseModel):
    ambient_temp_c: float | None = Field(default=None, ge=-20, le=50)
    device_model_wearable: str | None = None
    device_model_power: str | None = None
    device_model_cgm: str | None = None


class WearableSample(BaseModel):
    t_s: int = Field(ge=0, description="Seconds from session start.")
    hr_wearable_bpm: float | None = Field(default=None, ge=30, le=230)
    power_wearable_w: float | None = Field(default=None, ge=0, le=3000)
    cadence_rpm: float | None = Field(default=None, ge=0, le=200)
    cgm_glucose_mg_dl: float | None = Field(default=None, ge=30, le=400)


class PolicyOverride(BaseModel):
    rer_ceiling: float = Field(default=0.99, gt=0.7, le=1.1)
    lt2_prob_ceiling: float = Field(default=0.5, gt=0.0, lt=1.0)


class PredictRequest(BaseModel):
    subject: SubjectContext
    session: SessionContext = Field(default_factory=SessionContext)
    samples: list[WearableSample] = Field(min_length=1, max_length=MAX_SAMPLES)
    policy: PolicyOverride | None = None

    @model_validator(mode="after")
    def _t_s_strictly_increasing(self) -> PredictRequest:
        ts = [s.t_s for s in self.samples]
        if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
            raise ValueError("samples.t_s must be strictly increasing")
        if ts[-1] - ts[0] + 1 > MAX_SAMPLES:
            raise ValueError(f"session span exceeds {MAX_SAMPLES} seconds")
        return self


class SamplePrediction(BaseModel):
    t_s: int
    rer_pred: float
    vo2_pred_mlmin: float
    lt2_prob: float
    regime: str
    fat_ox_pred_g_per_min: float | None
    cho_ox_pred_g_per_min: float | None
    energy_pred_kcal_per_min: float | None


class PredictResponse(BaseModel):
    model_version: str
    n_samples: int
    n_grid_seconds: int = Field(description="Seconds in the 1 Hz grid the model scored.")
    imputed: dict[str, str]
    policy: dict[str, float]
    summary: dict[str, Any]
    samples: list[SamplePrediction]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_version: str
    bioml_version: str
    n_features: int
