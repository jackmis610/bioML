# Sensor-Noise Spec for Synthetic Wearable + CPET Generator

## 1. Optical (PPG) wrist HR vs ECG / chest-strap

### Cross-study summary by regime (wrist-worn, healthy adults)

| Regime | Bias (bpm) | MAE / RMSE (bpm) | LoA (95%) | Direction | Notes |
|---|---|---|---|---|---|
| Rest / sedentary | -0.3 to +1.0 | 1-3 | ±3-5 | minimal | Gillinov 2017, Bent 2020, Nelson & Allen 2019: rc 0.95-0.99 at rest |
| Low steady cycling (Z1-Z2) | -1 to +2 | 3-5 | ±6-8 | tends low | Gillinov 2017: rc 0.80-0.91 on cycle ergometer |
| Moderate steady cycling (Z3) | -2 to +3 | 4-6 | ±8-12 | tends low | Falter 2019 Apple Watch S4: MAE 3.9 bpm cycling moderate |
| High steady cycling (Z4-Z5) | -3 to +5 | 6-10 | ±10-15 | tends low | Pasadyn 2019: agreement deteriorates with HR >150 |
| Varying intensity / intervals | -5 to +10 | 8-15 | ±15-25 | bimodal | 5-15 s lag dominates error (Falter, Gillinov synthesis) |
| High-cadence cycling (>90 rpm) | spikes | 10-20 | ±25-40 | cadence lock possible | Running-Writings/Gillinov: PPG locks onto cadence harmonic when true HR ≈ cadence |

### Device differences

| Device | Best regime | Worst regime | Typical MAPE |
|---|---|---|---|
| Apple Watch S4-S9 (wrist) | Cycling moderate (MAE ~4) | Variable intensity (MAE 8-12) | 3-6% (Falter 2019, momentum.ai meta-analysis: bias -0.27, LoA -7.2 to +6.6) |
| Garmin Fenix/Forerunner (wrist) | Steady (MAPE 5%) | Trail/variable (MAPE 13%, ccc 0.32) | 5-13% |
| Whoop 4.0/5.0 (wrist) | Nocturnal (MAPE 3%) | Transient/HIIT (MAPE up to 15%) | 3-10% (MDPI 2025 transient-state study) |
| Fitbit Charge/Blaze | Treadmill | Cycling (rc 0.76 in Gillinov 2017) | 5-15% |
| Polar OH1 / Verity Sense (upper arm) | All exercise | Pure isometric | RMSE 1.6%, bias 0.27-0.33 bpm (Hettiarachchi 2019, PLoS One) |
| Whoop on bicep | High-intensity | --- | Best agreement with chest strap in placement study (PMC12788198) |
| Scosche Rhythm+ (forearm) | Cycling | --- | rc >0.80 cycling (Gillinov 2017) |

### Lag, dropout, artifact

- **Lag**: PPG trails ECG by **5-15 s** at HR transitions; smoothing window ~5-10 s. Steady-state lag ~0-2 s.
- **Dropout**: Real-world Apple Watch / Whoop workout files show **10-30 % missing seconds**, gaps typically ~30 s long, randomly distributed. Worse during high-intensity full-body motion.
- **Cadence lock probability**: not formally quantified in literature; runningwritings.com and Gillinov suggest occurs roughly when true HR is within ±10 bpm of cadence × 60 / pedal-stroke; estimate 5-15 % of cycling time-at-risk windows.
- **Skin tone**: Bent 2020 (npj Digital Medicine) found **no significant Fitzpatrick effect** in healthy controlled exercise. Earlier work and rebuttals suggest up to 15 % higher artifact rate in darker skin during motion; treat as a wide prior (+0-2 bpm RMSE for Fitzpatrick V-VI).

---

## 2. CGM (Dexcom G6/G7, FreeStyle Libre 2/3) vs venous/capillary

| Parameter | Dexcom G6 | Dexcom G7 | Libre 2 | Libre 3 |
|---|---|---|---|---|
| Overall MARD | ~9.0 % | 8.2 % arm / 9.1 % abdomen (Garg 2022 DT&T) | ~9.2 % | 7.9 % (Alva 2022) |
| Exercise MARD (aerobic) | 9-14 % | 8.9-13.9 % (Garg 2022) | 11-14 % | ~10-13 % (inferred) |
| Exercise MARD (HIIT) | up to 18 % | 7.7-14.5 % resistance, similar HIIT | up to 20 % (Moser 2019) | ~15 % |
| First-day MARD | ~12 % (degrades day 1) | ~9 % (G7 more stable than G6 day-1 mixed reports) | warm-up 1 h | warm-up 1 h |
| Steady-state MARD (days 3-7) | ~8 % | ~7.5 % | ~9 % | ~7.5 % |

### Lag

- **Physiologic ISF lag** at rest: **5-8 min** healthy; ~7-8 min T1D fasting.
- **During exercise**: **12 ± 11 min** lag during prolonged aerobic (Aberer et al. 2017); up to **20 min** during rapid drops; HIIT: large transient discordance — Moser 2016/2019 found CGM significantly differs from BG in 3/6 HIIT tests.
- **Direction during rapid drops** (e.g., HIIT recovery): CGM **overestimates** (lags above true). During rapid rises (post-CHO during exercise): CGM **underestimates**.

### Sensor-to-sensor variability (same person)

- Selvin 2023 (Clinical Chemistry, n=172, Dexcom G4 + Libre Pro paired): within-person CV for mean glucose **14-17 %**; CV for time-in-range **18-20 %**. Substantial discordance even at same timestamp.
- Across two simultaneous sensors of same model, expect typical mean-difference of 8-12 mg/dL and individual paired-point disagreement ±20 mg/dL.

### Compression artifact

Largely a sleep phenomenon; for cycling generator, can set probability ~0 during active periods, ~1-3 % per hour during pre/post rest if sensor on contact site.

---

## 3. Consumer VO2max estimates vs lab CPET

| Device | Bias | RMSE / MAPE | Confidence |
|---|---|---|---|
| Apple Watch S7 / S9 / Ultra 2 (Cardio Fitness, Cooper-style algo) | **−6.07 mL/kg/min** (S9/Ultra2, PLoS One 2025) | RMSE 8.85; MAPE 13-16 % | Independent validations diverge sharply from Apple's claim of ±1.2 mL/kg/min |
| Garmin (Firstbeat algo) | ~0 to −2 mL/kg/min (recreational), overestimates low-fit | margin ~3-3.5 mL/kg/min for ~95 % of users (vendor); external validations show MAPE ~8-12 % | Fair |
| Polar / Whoop | sparse independent data | Polar Fitness Test: SEE ~3-4 mL/kg/min vs CPET | Low |

Tendency across vendors: **overestimate** for low-fitness users, **underestimate** for elites. Heart-rate-reserve-based algos especially sensitive to HRmax estimation error.

---

## 4. Cycling power

| Device class | Vendor claim | Independent measured | Notes |
|---|---|---|---|
| Pedal-based (Favero Assioma, Garmin Vector 3) | ±1-2 % | CV <3 %, no sig. diff vs Quarq except at 500 W (Lillo-Bevia 2021; Pallarés 2018) | Most-validated class |
| Spider (Quarq DZero/DFour) | ±1.5 % | High agreement with Favero; not directly validated vs SRM in peer-reviewed work | Sound |
| Crank-arm dual (Stages LR, 4iiii Precision Pro) | ±1.5 % | Generally accepts ±2 % | Few peer-reviewed |
| Crank-arm single (Stages L, 4iiii Precision) | ±2 % | Doubles left → if 3 % L-heavy imbalance, **6 % overestimate** of total power; balance can drift with fatigue/sprints | Largest systematic bias category |
| Hub (PowerTap G3/GS) | ±1.5 % | Historically the reference; ±1-2 % consistent | Sound |
| Smart trainer Wahoo Kickr v5/v6 | ±1 % | DCR/independent: meets spec for steady state; ERG mode subject to inertia lag at transitions | Sound |
| Smart trainer Tacx Neo 2T/3M | <1 % | Hartog 2024: no sig. diff vs Vector 3 pedals; CV 0.03-0.15 % | Best-in-class |

### Drift

- Temperature drift causes zero-offset shift; ~2/3 of meters auto-compensate. Uncompensated: expect **0.5-2 % per 10 °C** without re-zero.
- Single-sided L/R imbalance bias: cyclists typically 47/53 to 53/47 → **±3-6 % systematic error** vs true total.

---

## 5. Operational signal-quality

- **PPG dropout in real workouts**: 10-30 % of seconds missing on wrist devices; bursts ~30 s. Lower (<5 %) at rest and on upper-arm placements.
- **Apple Watch HR sampling**: ~5 s intervals foreground (workout mode); ~1 s burst when needed. Aggregated to 1 Hz in export.
- **Garmin recording**: "Smart" recording subsamples (variable, mean ~3-5 s); "1 s" is true 1 Hz. Many newer devices default 1 s.
- **CGM sampling**: Dexcom G6 every 5 min; G7 every 5 min; Libre 2 every 1 min, Libre 3 every 1 min. No internal smoothing reported but G7 publishes smoothed predictive value.
- **GPS dropout**: open sky <1 % distance error; urban canyon ~1-3 % underestimation with occasional 5-20 m position errors; tree cover similar; tunnels = total loss until reacquisition (~10-30 s).
- **Smart-trainer ERG lag**: 1-3 s to settle on new target.
- **Timestamp drift**: ANT+/BLE devices typically within ±0.5 s of host; CGM phones drift up to a few seconds over hours. Apple/Garmin sync to NTP; fused stream needs ±1-2 s tolerance.
- **Battery downsample**: Garmin UltraTrac / power-save mode drops GPS sample to 60 s, HR to 60-120 s on long efforts; triggered by user setting, not auto by default on most recent devices. Apple Watch Low Power Mode reduces HR to every ~minute.

---

## 6. Running pace / cadence (brief)

Wrist GPS pace error: open course ~0.8 %, urban ~1.2 %, forest ~6 % distance error; instantaneous pace ±15-30 s/mi noise, ~3-5 s smoothing built in. Wrist cadence accurate to ±2-3 spm vs footpod.

---

## Starter Sensor-Noise Spec (YAML)

```yaml
# Per-second wearable observation layer. All Gaussian unless stated.

ppg_hr_wrist_appleish:        # Apple Watch / Garmin Fenix / Whoop wrist
  bias_mean_bpm: -0.5
  bias_sd_bpm: 1.5
  noise_sd_bpm:
    rest:        2.0
    z1_z2:       3.5
    z3:          5.0
    z4_z5:       7.5
    intervals:   10.0
    high_cadence_cycling: 12.0
  dropout_rate:                # P(missing | regime)
    rest:        0.02
    steady:      0.05
    high:        0.15
    intervals:   0.25
  dropout_burst_seconds: [10, 60]   # uniform length when a gap occurs
  lag_seconds_steady: 1.5
  lag_seconds_transient: 10.0       # additional during HR ramps
  smoothing_window_s: 7
  cadence_lock_probability_per_minute_cycling: 0.05  # at risk windows; conditional on |HR - cadence*k| < 10 bpm
  cadence_lock_duration_s: [10, 90]
  cadence_lock_offset_bpm: "+/- (cadence_rpm - true_hr)"
  spike_rate_per_minute: 0.02       # implausible ±30 bpm jumps
  flatline_rate_per_minute: 0.01    # HR stuck for >15 s
  skin_tone_factor_dark:            # multiplicative on noise_sd; wide prior
    mean: 1.10
    sd:   0.15

ppg_hr_upperarm_polarOH1_whoopBicep:
  bias_mean_bpm: 0.3
  bias_sd_bpm: 1.0
  noise_sd_bpm: {rest: 1.0, z1_z2: 1.5, z3: 2.0, z4_z5: 3.0, intervals: 4.5}
  dropout_rate: {rest: 0.005, steady: 0.01, high: 0.03, intervals: 0.05}
  lag_seconds_steady: 1.0
  lag_seconds_transient: 5.0
  cadence_lock_probability_per_minute_cycling: 0.005

chest_strap_ecg_polarH10:       # near-truth
  bias_mean_bpm: 0.0
  bias_sd_bpm: 0.3
  noise_sd_bpm: 0.8
  dropout_rate: 0.002
  lag_seconds: 0.0

cgm_dexcom_g7:
  mard_steady_pct: 8.2
  mard_exercise_aerobic_pct: 11.0
  mard_exercise_hiit_pct: 13.0
  noise_model: "multiplicative_lognormal"
  noise_cv_pct: 7.0               # day 3+ steady state
  noise_cv_day1_pct: 11.0
  lag_seconds_rest: 360
  lag_seconds_exercise: 720       # 12 min median during aerobic
  lag_seconds_hiit_drop: 900
  rapid_drop_bias_mgdl: +15       # CGM reads high during steepest drop
  rapid_rise_bias_mgdl: -10
  sampling_period_s: 300
  sensor_to_sensor_mean_diff_mgdl: 10
  sensor_to_sensor_tir_cv_pct: 19
  warmup_minutes: 30
  compression_low_probability_per_hour_at_rest: 0.02

cgm_libre3:
  mard_steady_pct: 7.9
  mard_exercise_pct: 12.0
  noise_cv_pct: 7.5
  lag_seconds_rest: 360
  lag_seconds_exercise: 720
  sampling_period_s: 60
  warmup_minutes: 60

vo2max_estimate_applewatch:
  bias_mean_mlkgmin: -6.0          # underestimates fit users
  bias_sd_mlkgmin: 3.5
  mape_pct: 14.0
  drift_per_week_mlkgmin: 0.3      # smoothing over rolling window
  fitness_dependent_bias:
    poor_fit: +3                   # overestimates low fit
    elite:    -8                   # underestimates elite

vo2max_estimate_garmin_firstbeat:
  bias_mean_mlkgmin: -1.0
  bias_sd_mlkgmin: 3.5
  mape_pct: 10.0

power_meter_dual_pedal_favero:
  scale_bias_mean_pct: 0.0
  scale_bias_sd_pct: 1.0
  noise_sd_pct_of_power: 1.5
  temp_drift_pct_per_10C: 0.4      # if temp-compensated
  zero_offset_jitter_pct: 0.3

power_meter_single_sided_stages_L:
  scale_bias_mean_pct: 0.0
  scale_bias_sd_pct: 3.0           # depends on rider L/R balance
  imbalance_bias_pct: "2 * (left_share - 0.5) * 100"   # systematic; sample left_share ~ N(0.50, 0.025)
  noise_sd_pct_of_power: 2.0
  temp_drift_pct_per_10C: 1.0      # many single-sided lack active compensation
  zero_offset_jitter_pct: 0.5

smart_trainer_kickr_v6:
  scale_bias_mean_pct: 0.0
  scale_bias_sd_pct: 1.0
  noise_sd_pct_of_power: 1.0
  erg_lag_seconds: 2.0

smart_trainer_tacx_neo_2t:
  scale_bias_mean_pct: 0.0
  scale_bias_sd_pct: 0.7
  noise_sd_pct_of_power: 0.8
  erg_lag_seconds: 1.5

gps_wrist:
  distance_error_pct_open_sky: 0.8
  distance_error_pct_urban: 2.0
  distance_error_pct_forest: 6.0
  instantaneous_pace_noise_s_per_km: 12
  dropout_probability_per_minute_tunnel: 1.0
  reacquisition_seconds: 20

system:
  inter_device_timestamp_drift_s_per_hour: 0.5
  device_sample_rates_hz:
    ppg_hr_export: 1.0
    chest_strap:   1.0
    power:         1.0
    cgm_dexcom:    0.00333          # 1/300
    cgm_libre:     0.01667          # 1/60
    gps:           1.0
  battery_save_downsample_after_hours: 6
  battery_save_hr_period_s: 60
```

### Citation table (short form)

- Gillinov 2017 — Med Sci Sports Exerc. PPG variability by modality; cycling rc 0.80-0.91, elliptical worst.
- Pasadyn 2019 — Cardiovasc Diagn Ther. Athlete treadmill; agreement deteriorates with motion/HR.
- Bent 2020 — npj Digital Medicine. Skin tone null effect; device × activity effects dominant.
- Nelson & Allen 2019 — Sensors. Multi-device validation across activity intensities.
- Falter 2019 — JMIR mHealth. Apple Watch S4 cycling MAE 3.9 bpm.
- Hettiarachchi 2019 — PLoS One. Polar OH1 RMSE 1.6 %.
- MDPI Sensors 2025 (PMC12568089) — WHOOP 4.0 transient-state errors.
- Garg 2022 — Diabetes Technol Ther. Dexcom G7 pivotal MARD.
- Alva 2022 — ADA abstract 76-LB. Libre 3 MARD 7.9 %.
- Aberer 2017; Moser 2019 — CGM exercise lag/HIIT accuracy.
- Selvin 2023 — Clin Chem. Within-person CGM variability.
- Lillo-Bevia & Pallarés 2018, 2021 — Favero validity vs Quarq.
- Hartog 2024 (PMC11348236) — Tacx Neo 2T vs Vector 3.
- PLoS One 2025 (PMC12080799) — Apple Watch VO2max bias −6.07.

### Confidence notes for the prior

- High confidence: PPG steady-state bias/LoA; chest-strap; Dexcom G7 baseline MARD; Favero/Tacx power.
- Medium: PPG dropout rates (mostly inferred from forum data + HRV-missingness studies); cadence-lock probabilities (described qualitatively).
- Low / widen priors: Whoop 5.0 (limited independent data — use Whoop 4.0 numbers ±50 %); skin-tone effect (Bent null vs earlier positive — keep as uncertain multiplier); CGM exercise lag direction during transitions (small N studies).
