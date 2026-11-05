# bioML Data Schema Design

The canonical data shape for synthetic generation and (eventually) real Aevox CPET + wearable data. Designed so the Aevox loader is a one-file swap: same schema, same downstream pipeline.

---

## Design principles

1. **Three-level hierarchy: subject → session → samples.** Mirrors the natural structure of CPET data.
2. **Truth and observation co-located.** Every sample row carries both lab-grade and wearable-grade signals. Real Aevox data will fill both (lab + watch); synthetic data fills both (simulated lab + degraded wearable).
3. **Nullable observations.** Wearable columns are nullable — that's how dropouts and missing-sensor scenarios are represented (e.g., a session without CGM has `cgm_glucose_mg_dl = null` throughout).
4. **Derived columns precomputed and nullable above LT2.** `fat_ox_g_per_min` and `cho_ox_g_per_min` are null in the Frayn-invalid range — never an interpolated or biased number.
5. **Breath-by-breath in, 1 Hz out.** CPET hardware exports breath-by-breath (per-exhalation rows of VO2/VCO2/RER/HR/VE). The loader resamples to 1 Hz by binning on second, linearly interpolating breath endpoints. Synthetic generator emits 1 Hz directly. The model sees only 1 Hz.
6. **Diet and glycogen are latent.** Real Aevox subjects won't reliably log fueling or training-load context. The model gets **no** diet-state or CHO-intake inputs from real data. The synthetic generator still simulates these internally (necessary for plausible RER drift) but hides them from the model, so the model learns to be robust to the marginal distribution. CGM, when present, provides indirect signal about fueling state.
7. **One session per subject is the default.** Matches the Aevox use case (one CPET per visit). Multi-session simulation remains available for sensitivity analyses but isn't the headline dataset shape.
8. **Wearable layer is device-heterogeneous.** Users bring their own watch / power meter / CGM. The synthetic generator randomizes device per subject across the noise profiles in `sensor_noise.md`, so the model trains on the full device-mix distribution.

---

## Tables

### `subjects` — one row per simulated or real subject

| Column | Type | Notes |
|---|---|---|
| subject_id | str (uuid) | primary key |
| age_years | int | |
| sex | enum {M, F} | |
| height_cm | float | |
| weight_kg | float | |
| ffm_kg | float | fat-free mass; for synthetic data, derived from sex + weight + body-fat assumption |
| training_status | enum {untrained, recreational, well_trained, elite} | for synthetic; for real Aevox, inferred from intake form or recent VO2max |
| habitual_diet | enum {mixed, high_carb, keto} | from intake form |
| vo2max_mlkgmin_true | float | **truth field** — what the lab CPET measured (or what the generator sampled) |
| hrmax_bpm_true | float | truth |
| hr_rest_bpm | float | observed |
| lt1_pct_vo2max_true | float | truth — generator-sampled or lab-detected from CPET |
| lt2_pct_vo2max_true | float | truth |
| fatmax_pct_vo2max_true | float | truth |
| mfo_g_per_min_true | float | truth |
| gross_efficiency_pct_true | float | truth |
| vo2_tau_moderate_s_true | float | truth (rarely measured in real CPETs; mostly synthetic-only) |
| initial_muscle_glycogen_mmol_per_kg_true | float | truth (never measured non-invasively; synthetic-only) |
| vo2max_estimated_wearable_mlkgmin | float (nullable) | what a Garmin/Apple Watch reports; available even when lab CPET hasn't been done |
| created_at | timestamp | |
| source | enum {synthetic_v1, aevox, public_dataset} | provenance |

**Notes:**
- Fields with `_true` suffix are the lab-side ground truth. The model never sees these at inference — they're targets / evaluation references.
- `vo2max_estimated_wearable_mlkgmin` IS available to the model — it's a noisy proxy for VO2max that a real wearable would report.
- For real Aevox data, `_true` fields are filled from CPET measurements; for synthetic data, from generator sampling. The schema doesn't distinguish.

### `sessions` — one row per CPET / training session

| Column | Type | Notes |
|---|---|---|
| session_id | str (uuid) | primary key |
| subject_id | str | foreign key |
| protocol | enum {ramp_incremental, step_incremental, steady_state_submax, long_endurance, intervals, free_ride} | |
| duration_s | int | total session length |
| diet_state_true | enum {fasted, fed_carb_2_3h, keto_adapted} (nullable) | **truth only** — synthetic-generator state, not a model input; null for real Aevox data |
| cho_intake_g_per_h_true | float (nullable) | **truth only** — synthetic; null for real |
| pre_session_glycogen_fraction_true | float (nullable) | **truth only** — synthetic-generator state; never measured in real subjects |
| ambient_temp_c | float (nullable) | observed where available |
| ambient_humidity_pct | float (nullable) | |
| device_model_wearable | str (nullable) | free-text e.g., "Apple Watch S9", "Garmin Fenix 7", "Whoop 5.0" — users bring their own |
| device_model_power | str (nullable) | e.g., "Favero Assioma Duo", "Wahoo Kickr v6", null if no power meter |
| device_model_cgm | str (nullable) | e.g., "Dexcom G7", null if no CGM |
| created_at | timestamp | |

### `samples` — one row per (session, second)

Lab-truth and wearable-observed columns side by side.

**Time + protocol**

| Column | Type | Notes |
|---|---|---|
| session_id | str | FK |
| subject_id | str | denormalized for filtering |
| t_s | int | seconds from session start |
| protocol_target_power_w | float (nullable) | what the test protocol called for; null in free-ride |

**Lab truth (the targets / eval reference)**

| Column | Type | Notes |
|---|---|---|
| power_lab_w | float | ergometer / SRM-grade reading |
| vo2_lab_mlmin | float | from metabolic cart, breath-resampled to 1 Hz |
| vco2_lab_mlmin | float | from metabolic cart |
| rer_lab | float | = vco2_lab / vo2_lab |
| hr_lab_bpm | float | chest-strap ECG-grade |
| pct_vo2max_lab | float | = vo2_lab / (vo2max_true × weight_kg), 0–~110 |
| ve_lab_lmin | float (nullable) | ventilation (sometimes available) |
| **Derived from truth** | | |
| fat_ox_g_per_min | float (nullable) | Frayn; null when `above_lt2_flag = true` |
| cho_ox_g_per_min | float (nullable) | Frayn; null when `above_lt2_flag = true` |
| above_lt2_flag | bool | true when `rer_lab ≥ 1.0` OR `pct_vo2max_lab > 85` |
| muscle_glycogen_mmol_per_kg_true | float | truth — synthetic only; rarely measurable in real data |
| **Optional truth** | | |
| blood_glucose_mg_dl | float (nullable) | fingerstick at fixed intervals; rare |
| blood_lactate_mmol_l | float (nullable) | fingerstick; ingest-able when present in Aevox sessions but not assumed |

**Wearable observations (the features the model sees)**

| Column | Type | Notes |
|---|---|---|
| hr_wearable_bpm | float (nullable) | PPG-derived; null on dropout seconds |
| power_wearable_w | float (nullable) | from rider's own power meter (different bias profile than lab) |
| cadence_rpm | float (nullable) | from speed/cadence sensor or power meter |
| cgm_glucose_mg_dl | float (nullable) | resampled to 1 Hz; same value held until next CGM reading (Dexcom = 5 min, Libre = 1 min) |
| accelerometer_movement_index | float (nullable) | optional; from watch IMU; rarely exported but plausible for Aevox device fusion |

**Metadata**

| Column | Type | Notes |
|---|---|---|
| is_warmup | bool | first ~3 min of session; useful for filtering |
| is_recovery | bool | last few minutes after main effort |

---

## Storage layout

```
data/
  synthetic/
    v1/                                # generator schema version 1
      subjects.parquet                 # ~10 KB per 1000 subjects
      sessions.parquet                 # ~50 KB per 1000 sessions
      samples/                         # Hive-partitioned parquet
        subject_id=<uuid>/
          session_id=<uuid>.parquet    # ~50–200 KB per session
  aevox/
    v1/
      subjects.parquet
      sessions.parquet
      samples/...                      # same layout, real data
  public/                              # if/when we add PhysioNet / Zenodo sets
    physionet_cpet/
      ...
```

**Why this layout:**
- Hive-style partitioning by `subject_id` enables subject-wise CV via cheap directory scans (no shuffle).
- Polars and DuckDB query partitioned Parquet natively.
- Per-session files mean we can stream-load big datasets without RAM blowup.
- Versioning the schema in the directory name (`v1/`) means schema migrations don't break old experiments.

**Estimated v1 dataset size:** 1,000 subjects × 1 session each × ~60 min average × 60 s × ~30 columns × 8 bytes ≈ **400 MB raw, ~80 MB Parquet-compressed**. Easy to keep in `data/` for the portfolio repo without git-LFS pain (gitignored, regenerable from seed). Scale up to 5,000 subjects if v1 is data-limited.

---

## Model I/O design

Inferred / decided from the schema, surfaced here for visibility.

### Features (model input — wearable-only)

Per second:
- `hr_wearable_bpm`, `power_wearable_w`, `cadence_rpm`, `cgm_glucose_mg_dl` (last-observation-carried-forward), optional `accelerometer_movement_index`
- Engineered: rolling stats over 10/30/60 s windows; HR-vs-power-mismatch indicator; `t_s` (time-in-session); time-since-last-CGM-update

Per session/subject (broadcast to every row):
- `age_years`, `sex`, `weight_kg`, `height_cm`, `ffm_kg`
- `vo2max_estimated_wearable_mlkgmin` (wearable's noisy estimate — NOT the lab truth)
- `ambient_temp_c` (when available)

**Deliberately NOT given to the model:** `diet_state_true`, `cho_intake_g_per_h_true`, `pre_session_glycogen_fraction_true`. These are latent in real Aevox data and the model must handle them via the marginal distribution. The CGM signal (when present) carries indirect fueling-state information that the model can learn to exploit.

### Targets (multitask)

- **Primary regression:** `vo2_lab_mlmin` (absolute VO2 in mL/min) — squared-error loss (revised 2026-11-05, decision D5: Huber at any δ in the tens of mL/min clipped gradients and under-predicted the top intensity bins)
- **Primary regression:** `rer_lab` — Huber loss δ=0.02
- **Auxiliary classification:** `above_lt2_flag` — BCE with positive-class weight ≈ 4

Predicting absolute VO2 alongside RER means downstream Frayn has both quantities it needs: `vco2_pred = rer_pred × vo2_pred`, then `fat_ox = 1.695·vo2_pred − 1.701·vco2_pred` (units L/min STPD; convert from mL/min as needed). VO2 is also independently valuable — "VO2 from wearables" is a competitive baseline for Apple/Garmin/Whoop.

### Downstream emission (inference-time)

For each second:
- If `lt2_flag_prob < 0.5` AND `rer_pred < 0.99`:
  - Emit `vo2_pred`, `rer_pred`, `fat_ox_pred`, `cho_ox_pred`, `regime="substrate_calculable"`
- Else:
  - Emit `vo2_pred`, `rer_pred`, `fat_ox=NA`, `cho_ox=NA`, `regime="buffering_dominated"`

### What the model never sees

Everything with the `_true` suffix in `subjects` and `samples`. These are the lab measurements / generator parameters — only used for label generation and evaluation. Strict train/eval discipline prevents accidental label leakage.

---

## What this commits us to (and what's deferred)

**Committed:**
- 1 Hz time resolution. Breath-by-breath data resampled at the loader.
- Three-level table structure with strict subject/session/sample separation.
- Parquet + Hive partitioning by `subject_id`.
- Multitask (VO2, RER, LT2-flag) model heads.
- Wearable-only features at inference. Lab features are for training labels and evaluation only.

**Deferred (out of v1 scope, easy to add later):**
- Treadmill / running modality — same schema, additional protocol values, but running economy differs and would need its own RER curve calibration.
- Multi-session longitudinal features (e.g., "training load over past 7 days" as a feature). Would require a `training_load` table.
- Lactate fingerstick measurements as additional truth-side data. Optional column already in schema, but not used by v1 generator or model.
- Inferring diet_state and CHO intake from CGM patterns rather than relying on user entry. A nice sub-project once v1 ships.
- Multi-device fusion (e.g., Apple Watch HR + Wahoo Tickr chest strap together). v1 assumes one wearable HR source per session.

---

## Aevox alignment — resolved 2026-05-21

| Question | Answer | Baked-in implication |
|---|---|---|
| CPET export format | Breath-by-breath | Loader resamples breath rows to 1 Hz; synthetic emits 1 Hz |
| Wearable data source | Users bring their own | `device_model_wearable` is free-text; generator randomizes device per subject |
| Blood lactate | Not routine but ingest-able | Optional nullable column; loader handles when present without requiring it |
| Nutrition / glycogen context | Too hard to gather | Treated as **latent**: synthetic generator simulates internally but model never sees diet/CHO/glycogen as input |
| Sessions per subject | One | Synthetic generator default: 1 session per subject; matches Aevox distribution |
