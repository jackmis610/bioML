"""Session-level sampling: protocol, power-time profile, diet state, devices.

v1 implements ramp_incremental (the canonical CPET protocol). Other protocols
will raise NotImplementedError until added.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.random import Generator

from bioml.schemas import DietState, HabitualDiet, Session, Subject

DIET_STATE_BY_HABITUAL: dict[HabitualDiet, dict[DietState, float]] = {
    "keto": {"keto_adapted": 1.0},
    "mixed": {"fasted": 0.55, "fed_carb_2_3h": 0.45},
    "high_carb": {"fasted": 0.20, "fed_carb_2_3h": 0.80},
}

WEARABLE_DEVICES = (
    "Apple Watch S9",
    "Garmin Fenix 7",
    "Whoop 5.0",
    "Polar OH1",
    "Garmin Forerunner 965",
    None,
)
POWER_DEVICES = (
    "Favero Assioma Duo",
    "Stages Power LR",
    "Quarq DZero",
    "Wahoo Kickr v6",
    "Tacx Neo 2T",
    None,
)
CGM_DEVICES = ("Dexcom G7", "FreeStyle Libre 3", None, None, None)

VO2_REST_ML_PER_KG_MIN = 3.5
KCAL_TO_ML_O2_AT_WATT = 2.87  # mL O2/min per 1 W of metabolic power


def _categorical(rng: Generator, choices: tuple[Any, ...]) -> Any:
    return choices[int(rng.integers(0, len(choices)))]


def _categorical_weighted(rng: Generator, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


def power_at_vo2max(subject: Subject) -> float:
    """External cycling power (W) required to elicit VO2max at this subject's GE."""
    vo2max_abs = subject.vo2max_mlkgmin_true * subject.weight_kg
    vo2_rest = VO2_REST_ML_PER_KG_MIN * subject.weight_kg
    ge_frac = subject.gross_efficiency_pct_true / 100
    return (vo2max_abs - vo2_rest) * ge_frac / KCAL_TO_ML_O2_AT_WATT


def ramp_power_profile(
    subject: Subject,
    rng: Generator,
    warmup_w: float = 50.0,
    warmup_s: int = 180,
    cooldown_s: int = 180,
) -> np.ndarray:
    """Power-time profile (W) for a ramp CPET.

    Rate is the subject's "preferred" rate by training status, then bounded by
    pmax so the ramp lasts 4-15 minutes and terminates near power-at-VO2max.
    """
    preferred = {
        "untrained": 15.0,
        "recreational": 20.0,
        "well_trained": 25.0,
        "elite": 30.0,
    }[subject.training_status]
    preferred += float(rng.normal(0, 2.0))

    pmax = power_at_vo2max(subject)
    rise = max(10.0, pmax - warmup_w)
    max_sensible_rate = rise / 4.0
    min_sensible_rate = rise / 15.0
    rate_w_per_min = max(min_sensible_rate, min(preferred, max_sensible_rate))
    rate_w_per_min = max(5.0, rate_w_per_min)

    ramp_duration_s = int(rise / rate_w_per_min * 60)
    ramp_duration_s = max(180, min(ramp_duration_s, 1200))

    n = warmup_s + ramp_duration_s + cooldown_s
    profile = np.empty(n, dtype=np.float64)
    profile[:warmup_s] = warmup_w
    t = np.arange(ramp_duration_s, dtype=np.float64)
    profile[warmup_s : warmup_s + ramp_duration_s] = warmup_w + (rate_w_per_min / 60) * t
    profile[warmup_s + ramp_duration_s :] = warmup_w
    return profile


def sample_diet_state(habitual_diet: HabitualDiet, rng: Generator) -> DietState:
    return _categorical_weighted(rng, DIET_STATE_BY_HABITUAL[habitual_diet])  # type: ignore[return-value]


def sample_session(
    subject: Subject,
    rng: Generator,
    protocol: str = "ramp_incremental",
) -> tuple[Session, np.ndarray]:
    """Sample a complete session (metadata + per-second power profile) for a subject."""
    if protocol != "ramp_incremental":
        raise NotImplementedError(f"protocol {protocol!r} not yet supported in v1")

    power = ramp_power_profile(subject, rng)
    duration_s = int(len(power))

    diet_state = sample_diet_state(subject.habitual_diet, rng)
    pre_glyc = float(rng.uniform(0.85, 1.0))
    ambient_temp = float(rng.normal(21.0, 2.0))

    session = Session(
        session_id=str(uuid.uuid4()),
        subject_id=subject.subject_id,
        protocol="ramp_incremental",
        duration_s=duration_s,
        diet_state_true=diet_state,
        cho_intake_g_per_h_true=0.0,
        pre_session_glycogen_fraction_true=round(pre_glyc, 3),
        ambient_temp_c=round(ambient_temp, 1),
        ambient_humidity_pct=None,
        device_model_wearable=_categorical(rng, WEARABLE_DEVICES),
        device_model_power=_categorical(rng, POWER_DEVICES),
        device_model_cgm=_categorical(rng, CGM_DEVICES),
        created_at=datetime.now(timezone.utc),
    )
    return session, power
