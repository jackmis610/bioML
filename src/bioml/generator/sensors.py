"""Wearable sensor noise overlay.

Degrades lab-truth signals into device-observed signals. Parameters come from
research/sensor_noise.md (PPG HR validation studies, CGM MARD, power meter
validation).

v1 implements: PPG HR (wrist + upper-arm profiles), power meter (5 device
classes), CGM (Dexcom G7 / Libre 3), cadence. v1 skips: accelerometer movement
index, GPS, temperature drift, battery-save downsampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bioml.schemas import Session

UPPER_ARM_DEVICES = ("Polar OH1",)


def simulate_cadence(power_w: NDArray[np.float64], rng: Generator) -> NDArray[np.float64]:
    """Cycling cadence (RPM). Rises mildly with power."""
    n = len(power_w)
    base = 70.0 + 0.05 * power_w
    base = np.clip(base, 60.0, 110.0)
    cadence = base + rng.normal(0.0, 4.0, n)
    return np.clip(cadence, 50.0, 130.0)


def simulate_power_meter(
    power_truth_w: NDArray[np.float64],
    device: str | None,
    rng: Generator,
) -> NDArray[np.float64]:
    """Observed power from a cycling power meter."""
    n = len(power_truth_w)
    if device is None:
        return np.full(n, np.nan)

    profile = {
        "Favero Assioma Duo": (1.0, 1.5),
        "Quarq DZero": (1.0, 1.5),
        "Stages Power LR": (2.0, 2.0),
        "Wahoo Kickr v6": (1.0, 1.0),
        "Tacx Neo 2T": (0.7, 0.8),
    }
    scale_bias_sd_pct, noise_sd_pct = profile.get(device, (1.5, 1.5))

    scale_bias = 1.0 + rng.normal(0.0, scale_bias_sd_pct / 100.0)
    noise = rng.normal(0.0, 1.0, n) * (noise_sd_pct / 100.0) * power_truth_w
    return scale_bias * power_truth_w + noise


def _intensity_regime(pct_vo2max: float) -> str:
    if pct_vo2max < 30:
        return "rest"
    if pct_vo2max < 60:
        return "z1z2"
    if pct_vo2max < 80:
        return "z3"
    return "z4z5"


def _dropout_regime(pct_vo2max: float) -> str:
    if pct_vo2max < 30:
        return "rest"
    if pct_vo2max < 60:
        return "steady"
    if pct_vo2max < 85:
        return "high"
    return "intervals"


def simulate_ppg_hr(
    hr_truth_bpm: NDArray[np.float64],
    pct_vo2max: NDArray[np.float64],
    cadence_rpm: NDArray[np.float64],
    device: str | None,
    rng: Generator,
) -> NDArray[np.float64]:
    """Observed wrist or upper-arm PPG HR. NaN on dropout seconds."""
    n = len(hr_truth_bpm)
    if device is None:
        return np.full(n, np.nan)

    is_upper_arm = device in UPPER_ARM_DEVICES

    if is_upper_arm:
        bias_mean, bias_sd = 0.3, 1.0
        noise_sd_table = {"rest": 1.0, "z1z2": 1.5, "z3": 2.0, "z4z5": 3.0}
        dropout_target = {"rest": 0.005, "steady": 0.01, "high": 0.03, "intervals": 0.05}
        cadence_lock_per_min = 0.005
    else:
        bias_mean, bias_sd = -0.5, 1.5
        noise_sd_table = {"rest": 2.0, "z1z2": 3.5, "z3": 5.0, "z4z5": 7.5}
        dropout_target = {"rest": 0.02, "steady": 0.05, "high": 0.15, "intervals": 0.25}
        cadence_lock_per_min = 0.05

    bias = float(rng.normal(bias_mean, bias_sd))

    noise_sd_arr = np.array([noise_sd_table[_intensity_regime(p)] for p in pct_vo2max])
    noisy = hr_truth_bpm + bias + rng.normal(0.0, 1.0, n) * noise_sd_arr

    # 7-second causal moving average (typical manufacturer smoothing)
    smooth_window = 7
    smoothed = np.copy(noisy)
    if n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed[smooth_window - 1 :] = np.convolve(noisy, kernel, mode="valid")

    # Dropouts: target steady-state fraction missing per regime, ~30s average burst
    avg_burst = 30.0
    start_prob_arr = np.array([dropout_target[_dropout_regime(p)] / avg_burst for p in pct_vo2max])
    dropout_mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if rng.random() < start_prob_arr[i]:
            burst = int(rng.uniform(10, 60))
            dropout_mask[i : min(i + burst, n)] = True
            i += burst
        else:
            i += 1

    # Cadence lock: chance per minute when |HR - cadence| < 10 bpm during cycling
    lock_per_sec = cadence_lock_per_min / 60.0
    j = 0
    while j < n:
        if (
            not dropout_mask[j]
            and abs(hr_truth_bpm[j] - cadence_rpm[j]) < 10.0
            and rng.random() < lock_per_sec
        ):
            lock_dur = int(rng.uniform(10, 90))
            end = min(j + lock_dur, n)
            smoothed[j:end] = cadence_rpm[j:end] + rng.normal(0.0, 2.0, end - j)
            j = end
            continue
        j += 1

    smoothed[dropout_mask] = np.nan
    return smoothed


def simulate_cgm(
    diet_state: str | None,
    duration_s: int,
    pct_vo2max: NDArray[np.float64],
    device: str | None,
    rng: Generator,
) -> NDArray[np.float64]:
    """Observed CGM glucose at 1 Hz, last-value-carried-forward between samples."""
    if device is None:
        return np.full(duration_s, np.nan)

    baseline = {"fasted": 85.0, "fed_carb_2_3h": 110.0, "keto_adapted": 75.0}.get(
        diet_state or "fasted", 90.0
    )
    sample_period_s = 60 if device == "FreeStyle Libre 3" else 300
    n_samples = duration_s // sample_period_s + 2

    samples = np.empty(n_samples)
    samples[0] = baseline + float(rng.normal(0, 5))
    for i in range(1, n_samples):
        mean_revert = (baseline - samples[i - 1]) * 0.08
        t_window_start = max(0, i * sample_period_s - sample_period_s)
        t_window_end = min(duration_s, (i + 1) * sample_period_s)
        avg_pct = (
            float(np.mean(pct_vo2max[t_window_start:t_window_end]))
            if t_window_end > t_window_start
            else 0.0
        )
        exercise_drift = -1.5 if avg_pct > 50 else 0.0
        samples[i] = samples[i - 1] + mean_revert + exercise_drift + float(rng.normal(0, 2.0))

    mard_pct = 0.10 if float(np.mean(pct_vo2max > 60)) > 0.3 else 0.08
    samples = samples * (1.0 + rng.normal(0.0, mard_pct / 1.96, n_samples))

    return np.repeat(samples, sample_period_s)[:duration_s]


def apply_sensor_overlay(
    truth: dict[str, NDArray[Any]],
    session: Session,
    rng: Generator,
) -> dict[str, NDArray[Any]]:
    """Compute all wearable observation columns from lab-truth time series."""
    pct = truth["pct_vo2max_lab"]
    hr_truth = truth["hr_lab_bpm"]
    power_truth = truth["power_lab_w"]
    duration_s = len(pct)

    cadence = simulate_cadence(power_truth, rng)
    hr_w = simulate_ppg_hr(hr_truth, pct, cadence, session.device_model_wearable, rng)
    power_w = simulate_power_meter(power_truth, session.device_model_power, rng)
    cgm = simulate_cgm(session.diet_state_true, duration_s, pct, session.device_model_cgm, rng)

    return {
        "hr_wearable_bpm": hr_w,
        "power_wearable_w": power_w,
        "cadence_rpm": cadence,
        "cgm_glucose_mg_dl": cgm,
        "accelerometer_movement_index": np.full(duration_s, np.nan),
    }
