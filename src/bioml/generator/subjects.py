"""Subject-level prior sampling.

Draws subject parameters from the priors in research/spec.yaml. Encodes the
Q1-resolved bivariate-Gaussian sampling for (LT1, LT2) with r=0.60 and the
LT2 - LT1 >= 8 pp floor, and the lognormal MFO prior.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.random import Generator

from bioml.config import stratum_value
from bioml.schemas import HabitualDiet, Sex, Subject, TrainingStatus

HABITUAL_DIET_WEIGHTS = {"mixed": 0.85, "high_carb": 0.05, "keto": 0.10}


def _truncated_normal(
    rng: Generator,
    mean: float,
    sd: float,
    low: float,
    high: float,
    max_tries: int = 100,
) -> float:
    for _ in range(max_tries):
        val = rng.normal(mean, sd)
        if low <= val <= high:
            return float(val)
    return float(np.clip(rng.normal(mean, sd), low, high))


def _bivariate_truncated(
    rng: Generator,
    means: tuple[float, float],
    sds: tuple[float, float],
    rho: float,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    min_gap: float,
    max_tries: int = 200,
) -> tuple[float, float]:
    m1, m2 = means
    s1, s2 = sds
    cov_matrix = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    (lo1, hi1), (lo2, hi2) = bounds
    for _ in range(max_tries):
        x, y = rng.multivariate_normal([m1, m2], cov_matrix)
        if lo1 <= x <= hi1 and lo2 <= y <= hi2 and (y - x) >= min_gap:
            return float(x), float(y)
    x = float(np.clip(rng.normal(m1, s1), lo1, hi1))
    y = float(np.clip(rng.normal(m2, s2), max(lo2, x + min_gap), hi2))
    return x, y


def _categorical(rng: Generator, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs = probs / probs.sum()
    idx = rng.choice(len(keys), p=probs)
    return keys[idx]


def sample_subject(spec: dict[str, Any], rng: Generator) -> Subject:
    """Draw one synthetic subject's true physiological parameters."""
    subj = spec["subject"]

    age = int(rng.integers(subj["age"]["bounds"][0], subj["age"]["bounds"][1] + 1))
    sex: Sex = "M" if rng.random() < subj["sex_M_probability"] else "F"
    training_status: TrainingStatus = _categorical(rng, subj["training_status"]["weights"])  # type: ignore[assignment]
    habitual_diet: HabitualDiet = _categorical(rng, HABITUAL_DIET_WEIGHTS)  # type: ignore[assignment]

    vo2_cfg = subj["vo2max_mlkgmin"]
    vo2_params = stratum_value(vo2_cfg, training_status, sex)
    vo2 = _truncated_normal(rng, vo2_params[0], vo2_params[1], *vo2_cfg["bounds"])
    if age > 25:
        dec_mean, dec_sd = vo2_cfg["age_decrement_per_year"]
        slope = _truncated_normal(rng, dec_mean, dec_sd, 0.05, 1.0)
        vo2 = max(vo2_cfg["bounds"][0], vo2 - slope * (age - 25))

    hrmax_cfg = subj["hrmax_bpm"]
    hrmax = _truncated_normal(rng, 208 - 0.7 * age, hrmax_cfg["residual_sd"], *hrmax_cfg["bounds"])

    hr_rest_cfg = subj["hr_rest_bpm"]
    hr_rest_params = stratum_value(hr_rest_cfg, training_status, sex)
    hr_rest = _truncated_normal(rng, hr_rest_params[0], hr_rest_params[1], *hr_rest_cfg["bounds"])

    th_cfg = subj["thresholds_pct_vo2max"]
    th_strat = th_cfg["by_stratum"][training_status]
    lt1, lt2 = _bivariate_truncated(
        rng,
        means=(th_strat["lt1"][0], th_strat["lt2"][0]),
        sds=(th_strat["lt1"][1], th_strat["lt2"][1]),
        rho=th_cfg["lt1_lt2_correlation"],
        bounds=(tuple(th_cfg["bounds"]["lt1"]), tuple(th_cfg["bounds"]["lt2"])),
        min_gap=8.0,
    )

    fm_cfg = subj["fatmax_pct_vo2max"]
    fm_params = stratum_value(fm_cfg, training_status, sex)
    fatmax_hi = max(fm_cfg["bounds"][0] + 1, lt1 - 2.0)
    fatmax = _truncated_normal(rng, fm_params[0], fm_params[1], fm_cfg["bounds"][0], fatmax_hi)

    mfo_cfg = subj["mfo_g_per_min"]
    mfo_strat = mfo_cfg["by_stratum"][training_status]
    if isinstance(mfo_strat, dict) and "mu_log" not in mfo_strat:
        mfo_params = mfo_strat[sex]
    else:
        mfo_params = mfo_strat
    mfo = float(np.exp(rng.normal(mfo_params["mu_log"], mfo_params["sigma_log"])))
    if habitual_diet == "keto":
        mfo *= mfo_cfg["diet_multiplier_keto"]
    mfo = float(np.clip(mfo, *mfo_cfg["bounds"]))

    ge_cfg = subj["gross_efficiency_pct"]
    ge_params = stratum_value(ge_cfg, training_status)
    ge = _truncated_normal(rng, ge_params[0], ge_params[1], *ge_cfg["bounds"])

    tau_cfg = subj["vo2_tau_moderate_s"]
    tau_params = stratum_value(tau_cfg, training_status)
    tau = _truncated_normal(rng, tau_params[0], tau_params[1], *tau_cfg["bounds"])

    glyc_cfg = subj["initial_muscle_glycogen_mmol_per_kg"]
    glyc_params = stratum_value(glyc_cfg, training_status)
    glycogen = _truncated_normal(rng, glyc_params[0], glyc_params[1], *glyc_cfg["bounds"])

    height = _truncated_normal(rng, 178 if sex == "M" else 165, 7, 150, 200)
    base_weight = 78 if sex == "M" else 64
    weight = _truncated_normal(rng, base_weight, 11, 45, 130)
    bf_mean = 18 if sex == "M" else 26
    if training_status in ("well_trained", "elite"):
        bf_mean -= 5
    body_fat_pct = _truncated_normal(rng, bf_mean, 4, 6 if sex == "M" else 14, 40)
    ffm = max(25.5, weight * (1 - body_fat_pct / 100))

    vo2_estimate = vo2 + _truncated_normal(rng, -1.0, 3.5, -15, 15)

    return Subject(
        subject_id=str(uuid.uuid4()),
        age_years=age,
        sex=sex,
        height_cm=round(height, 1),
        weight_kg=round(weight, 1),
        ffm_kg=round(ffm, 1),
        training_status=training_status,
        habitual_diet=habitual_diet,
        vo2max_mlkgmin_true=round(vo2, 2),
        hrmax_bpm_true=round(hrmax, 1),
        hr_rest_bpm=round(hr_rest, 1),
        lt1_pct_vo2max_true=round(lt1, 1),
        lt2_pct_vo2max_true=round(lt2, 1),
        fatmax_pct_vo2max_true=round(fatmax, 1),
        mfo_g_per_min_true=round(mfo, 3),
        gross_efficiency_pct_true=round(ge, 2),
        vo2_tau_moderate_s_true=round(tau, 1),
        initial_muscle_glycogen_mmol_per_kg_true=round(glycogen, 1),
        vo2max_estimated_wearable_mlkgmin=round(vo2_estimate, 2),
        created_at=datetime.now(UTC),
        source="synthetic_v1",
    )


def sample_subjects(spec: dict[str, Any], n: int, seed: int = 42) -> list[Subject]:
    rng = np.random.default_rng(seed)
    return [sample_subject(spec, rng) for _ in range(n)]
