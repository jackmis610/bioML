"""Per-second physiology simulation: VO2, RER, glycogen, HR for one session.

Encodes the Q2 (3-segment piecewise-linear RER curve) and Q3 (biphasic-linear
glycogen depletion with intensity scaling) design decisions. v1 skips:
- VO2 slow component above LT2 (adds VO2 drift; flagged in spec)
- HR cardiac drift over time at constant intensity
- Blood-glucose channel from exogenous CHO intake
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bioml.frayn import cho_oxidation_g_per_min, fat_oxidation_g_per_min
from bioml.generator.sessions import VO2_REST_ML_PER_KG_MIN
from bioml.schemas import Session, Subject

DIET_OFFSETS = {
    "fasted": 0.0,
    "fed_carb_2_3h": 0.04,
    "keto_adapted": -0.06,
}


def steady_state_vo2_mlmin(power_w: float, weight_kg: float, ge_pct: float) -> float:
    """VO2 (mL/min) at steady state for a given cycling power."""
    vo2_rest = VO2_REST_ML_PER_KG_MIN * weight_kg
    return vo2_rest + 287.0 * power_w / ge_pct


def rer_curve(
    pct_vo2max: NDArray[np.float64] | float,
    rest_rer: float,
    fatmax_rer: float,
    lt1_rer: float,
    lt2_rer: float,
    max_rer: float,
    fatmax_pct: float,
    lt1: float,
    lt2: float,
) -> NDArray[np.float64]:
    """4-segment piecewise-linear RER curve. Knees at Fatmax, LT1, LT2.

    The Fatmax knee is what produces the bell-shaped fat oxidation curve:
    below Fatmax the RER slope is shallow (s0 < critical slope -> fat_ox rising);
    above Fatmax the slope steepens through s1 -> s2 -> s3 (fat_ox falling).
    """
    x = np.asarray(pct_vo2max, dtype=np.float64)
    s0 = (fatmax_rer - rest_rer) / fatmax_pct
    s1 = (lt1_rer - fatmax_rer) / (lt1 - fatmax_pct)
    s2 = (lt2_rer - lt1_rer) / (lt2 - lt1)
    s3 = (max_rer - lt2_rer) / (100.0 - lt2)
    return np.where(
        x < fatmax_pct,
        rest_rer + s0 * x,
        np.where(
            x < lt1,
            fatmax_rer + s1 * (x - fatmax_pct),
            np.where(
                x < lt2,
                lt1_rer + s2 * (x - lt1),
                lt2_rer + s3 * (x - lt2),
            ),
        ),
    )


def glycogen_depletion_rate_mmol_per_kg_per_min(
    pct_vo2max: float,
    glycogen_current: float,
    glycogen_initial: float,
    r1_at_65: float,
    knee_fraction: float,
    phase2_multiplier: float,
) -> float:
    """Instantaneous glycogen depletion rate."""
    if pct_vo2max < 40.0:
        return 0.0
    pct = min(pct_vo2max, 90.0)
    r = float(r1_at_65 * (pct / 65.0) ** 1.8)
    if glycogen_current < knee_fraction * glycogen_initial:
        r *= phase2_multiplier
    return r


def run_session(
    subject: Subject,
    session: Session,
    power_profile_w: NDArray[np.float64],
    spec: dict[str, Any],
    rng: Generator,
) -> dict[str, NDArray[Any]]:
    """Simulate one full session. Returns per-second arrays keyed by Sample column name."""
    n = len(power_profile_w)
    weight = subject.weight_kg
    ge_pct = subject.gross_efficiency_pct_true
    tau = subject.vo2_tau_moderate_s_true
    vo2max_abs = subject.vo2max_mlkgmin_true * weight
    vo2_rest = VO2_REST_ML_PER_KG_MIN * weight
    hr_rest = subject.hr_rest_bpm
    hr_max = subject.hrmax_bpm_true
    lt1 = subject.lt1_pct_vo2max_true
    lt2 = subject.lt2_pct_vo2max_true
    fatmax_pct = subject.fatmax_pct_vo2max_true

    rer_anchors = spec["rer_curve"]["anchor_rer"][subject.training_status]
    breath_noise_sd = spec["rer_curve"]["breath_noise_sd_30s_window"]

    diet_offset = (
        DIET_OFFSETS.get(session.diet_state_true, 0.0)
        if session.diet_state_true is not None
        else 0.0
    )

    glyc_cfg = spec["glycogen"]
    r1_at_65_params = glyc_cfg["r1_at_65pct_vo2max_mmol_per_kg_per_min"]["by_stratum"][
        subject.training_status
    ]
    r1_at_65 = float(
        rng.normal(r1_at_65_params[0], r1_at_65_params[1])
        * np.exp(rng.normal(0, glyc_cfg["noise_log_sigma"]))
    )
    r1_at_65 = max(0.1, r1_at_65)
    knee_fraction = glyc_cfg["knee_fraction_of_G0"][subject.training_status]
    phase2_mult = glyc_cfg["phase2_rate_multiplier"]

    drift_beta = spec["rer_drift"]["beta_by_stratum"][subject.training_status]

    g0 = subject.initial_muscle_glycogen_mmol_per_kg_true
    pre_glyc_frac = (
        session.pre_session_glycogen_fraction_true
        if session.pre_session_glycogen_fraction_true is not None
        else 1.0
    )
    glycogen = g0 * pre_glyc_frac

    alpha = 1.0 - np.exp(-1.0 / tau)

    vo2 = np.empty(n)
    pct_arr = np.empty(n)
    rer_arr = np.empty(n)
    vco2 = np.empty(n)
    hr_arr = np.empty(n)
    glyc_arr = np.empty(n)
    fat_ox_arr = np.full(n, np.nan)
    cho_ox_arr = np.full(n, np.nan)
    above_lt2_arr = np.zeros(n, dtype=bool)

    vo2_current = vo2_rest
    noise_per_sec_sd = breath_noise_sd / np.sqrt(30.0)

    for t in range(n):
        vo2_ss = steady_state_vo2_mlmin(power_profile_w[t], weight, ge_pct)
        vo2_ss = min(vo2_ss, vo2max_abs * 1.06)
        vo2_current = vo2_current + (vo2_ss - vo2_current) * alpha
        vo2[t] = vo2_current

        pct = (vo2_current - vo2_rest) / (vo2max_abs - vo2_rest) * 100.0
        pct = float(np.clip(pct, 0.0, 110.0))
        pct_arr[t] = pct

        base_rer = float(
            rer_curve(
                np.array([pct], dtype=np.float64),
                rer_anchors["rest"],
                rer_anchors["fatmax"],
                rer_anchors["lt1"],
                rer_anchors["lt2"],
                rer_anchors["max"],
                fatmax_pct,
                lt1,
                lt2,
            )[0]
        )
        g_frac = glycogen / g0 if g0 > 0 else 0.0
        drift = -drift_beta * (1.0 - g_frac)
        r_total = base_rer + drift + diet_offset + float(rng.normal(0, noise_per_sec_sd))
        r_total = float(np.clip(r_total, 0.65, 1.25))
        rer_arr[t] = r_total

        vco2_t = r_total * vo2_current
        vco2[t] = vco2_t

        hr_t = hr_rest + (hr_max - hr_rest) * (vo2_current - vo2_rest) / (vo2max_abs - vo2_rest)
        hr_t = float(np.clip(hr_t, hr_rest, hr_max * 1.02))
        hr_arr[t] = hr_t

        above = r_total >= 1.0 or pct > 85.0
        above_lt2_arr[t] = above

        if not above and 0.71 <= r_total <= 1.0:
            vo2_l = vo2_current / 1000.0
            vco2_l = vco2_t / 1000.0
            fat_ox_arr[t] = float(fat_oxidation_g_per_min(np.array([vo2_l]), np.array([vco2_l]))[0])
            cho_ox_arr[t] = float(
                cho_oxidation_g_per_min(np.array([vo2_l]), np.array([vco2_l]), regime="exercise")[0]
            )

        depletion = glycogen_depletion_rate_mmol_per_kg_per_min(
            pct, glycogen, g0, r1_at_65, knee_fraction, phase2_mult
        )
        glycogen = max(0.0, glycogen - depletion / 60.0)
        glyc_arr[t] = glycogen

    return {
        "t_s": np.arange(n),
        "power_lab_w": power_profile_w,
        "vo2_lab_mlmin": vo2,
        "vco2_lab_mlmin": vco2,
        "rer_lab": rer_arr,
        "hr_lab_bpm": hr_arr,
        "pct_vo2max_lab": pct_arr,
        "fat_ox_g_per_min": fat_ox_arr,
        "cho_ox_g_per_min": cho_ox_arr,
        "above_lt2_flag": above_lt2_arr,
        "muscle_glycogen_mmol_per_kg_true": glyc_arr,
    }
