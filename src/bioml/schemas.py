"""Pydantic schemas matching research/schema_design.md.

Three-level hierarchy: Subject -> Session -> Sample. Truth columns (suffix
`_true`) come from lab CPET or generator state and are quarantined from
model inputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TrainingStatus = Literal["untrained", "recreational", "well_trained", "elite"]
Sex = Literal["M", "F"]
HabitualDiet = Literal["mixed", "high_carb", "keto"]
DietState = Literal["fasted", "fed_carb_2_3h", "keto_adapted"]
Protocol = Literal[
    "ramp_incremental",
    "step_incremental",
    "steady_state_submax",
    "long_endurance",
    "intervals",
    "free_ride",
]
DataSource = Literal["synthetic_v1", "aevox", "public_dataset"]


class Subject(BaseModel):
    subject_id: str
    age_years: int = Field(ge=18, le=99)
    sex: Sex
    height_cm: float = Field(gt=120, lt=230)
    weight_kg: float = Field(gt=35, lt=200)
    ffm_kg: float = Field(gt=25, lt=150)
    training_status: TrainingStatus
    habitual_diet: HabitualDiet

    vo2max_mlkgmin_true: float = Field(ge=15, le=100)
    hrmax_bpm_true: float = Field(ge=140, le=230)
    hr_rest_bpm: float = Field(ge=28, le=110)
    lt1_pct_vo2max_true: float = Field(ge=25, le=85)
    lt2_pct_vo2max_true: float = Field(ge=50, le=95)
    fatmax_pct_vo2max_true: float = Field(ge=20, le=85)
    mfo_g_per_min_true: float = Field(ge=0.05, le=2.5)
    gross_efficiency_pct_true: float = Field(ge=15, le=28)
    vo2_tau_moderate_s_true: float = Field(ge=4, le=70)
    initial_muscle_glycogen_mmol_per_kg_true: float = Field(ge=50, le=220)

    vo2max_estimated_wearable_mlkgmin: float | None = None
    created_at: datetime
    source: DataSource


class Session(BaseModel):
    session_id: str
    subject_id: str
    protocol: Protocol
    duration_s: int = Field(gt=0)

    diet_state_true: DietState | None = None
    cho_intake_g_per_h_true: float | None = Field(default=None, ge=0, le=200)
    pre_session_glycogen_fraction_true: float | None = Field(default=None, ge=0, le=1)

    ambient_temp_c: float | None = None
    ambient_humidity_pct: float | None = Field(default=None, ge=0, le=100)
    device_model_wearable: str | None = None
    device_model_power: str | None = None
    device_model_cgm: str | None = None
    created_at: datetime


class Sample(BaseModel):
    session_id: str
    subject_id: str
    t_s: int = Field(ge=0)

    protocol_target_power_w: float | None = Field(default=None, ge=0)

    power_lab_w: float = Field(ge=0)
    vo2_lab_mlmin: float = Field(ge=0)
    vco2_lab_mlmin: float = Field(ge=0)
    rer_lab: float = Field(ge=0.5, le=1.5)
    hr_lab_bpm: float = Field(ge=30, le=230)
    pct_vo2max_lab: float = Field(ge=0, le=130)
    ve_lab_lmin: float | None = Field(default=None, ge=0)

    fat_ox_g_per_min: float | None = Field(default=None, ge=0)
    cho_ox_g_per_min: float | None = Field(default=None, ge=0)
    above_lt2_flag: bool
    muscle_glycogen_mmol_per_kg_true: float = Field(ge=0)

    blood_glucose_mg_dl: float | None = Field(default=None, ge=30, le=400)
    blood_lactate_mmol_l: float | None = Field(default=None, ge=0, le=25)

    hr_wearable_bpm: float | None = Field(default=None, ge=30, le=230)
    power_wearable_w: float | None = Field(default=None, ge=0)
    cadence_rpm: float | None = Field(default=None, ge=0, le=200)
    cgm_glucose_mg_dl: float | None = Field(default=None, ge=30, le=400)
    accelerometer_movement_index: float | None = Field(default=None, ge=0)

    is_warmup: bool = False
    is_recovery: bool = False
