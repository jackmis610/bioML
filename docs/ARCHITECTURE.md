# bioML architecture

**One sentence.** bioML predicts what a metabolic cart measures (VO2, RER, fat and carbohydrate oxidation) from what a consumer wearable records (PPG heart rate, optional power meter, optional CGM, intake-form context), using a physiology-grounded synthetic dataset as the phase-1 training ground and a schema that lets real CPET data drop in without refactoring.

This document is the walk-through: what each layer does, why it is built that way, what it costs, and where it is weak. Every number quoted here is regenerated from `reports/v2/metrics.json` by `scripts/sync_docs.py`; nothing is typed by hand.

Companion documents: [`research/design_decisions.md`](../research/design_decisions.md) (the four load-bearing physiology calls), [`research/schema_design.md`](../research/schema_design.md) (table contracts), [`research/physiology_priors.md`](../research/physiology_priors.md) and [`research/sensor_noise.md`](../research/sensor_noise.md) (every prior with its citation), and [`reports/v2/REPORT.md`](../reports/v2/REPORT.md) (the full evaluation).

---

## 1. The problem and the framing

A cardiopulmonary exercise test (CPET) is the gold standard for measuring how a person burns fuel during exercise: a metabolic cart measures oxygen consumed (VO2) and carbon dioxide produced (VCO2) breath by breath, and the ratio RER = VCO2/VO2 encodes the fat/carbohydrate mix (RER 0.71 is pure fat, 1.00 is pure carbohydrate). From VO2 and VCO2 the Frayn stoichiometry gives grams per minute of each fuel. The outputs that matter to an athlete or a metabolic-health program are the **Fatmax zone** (the intensity at which fat oxidation peaks), the **thresholds** (LT1, LT2) that bound training zones, and substrate use over a session.

CPETs are expensive, lab-bound, and done once. Wearables are cheap, continuous, and noisy. The engineering question is: **how much of the CPET output can be recovered from wearable signals alone, and how do you know when the answer is not trustworthy?**

Three framing decisions shape everything downstream:

1. **Wearable-only inputs, lab-only labels.** Every column with a `_lab` or `_true` suffix is quarantined from the feature builder and tested to stay out. If the model could see lab VO2 it would be a lookup table.
2. **Predict the smooth physiological quantities, then apply the chemistry downstream.** The model predicts RER and VO2 across the whole intensity range. Frayn is applied at inference only where it is valid, with an explicit `regime` label otherwise. This is design decision D4 and it is the single most important modeling choice in the project (section 7).
3. **Synthetic first, real second, same contract.** No real CPET-plus-wearable dataset exists yet (the Aevox partnership supplies it in phase 2). Phase 1 builds a generator whose every parameter traces to a citation, and designs the data contract so the real loader is a one-file swap.

---

## 2. System at a glance

```mermaid
flowchart LR
    subgraph research["research/ (source of truth)"]
        spec[spec.yaml<br/>priors + sensor noise + decisions]
    end

    subgraph gen["Synthetic generator  (bioml.generator)"]
        subj[subjects.py<br/>sample physiology] --> sess[sessions.py<br/>ramp CPET protocol]
        sess --> phys[physiology.py<br/>per-second VO2 / RER / HR / glycogen]
        phys --> sens[sensors.py<br/>PPG, power meter, CGM degradation]
    end
    spec --> subj

    sens --> parquet[(data/synthetic/v1<br/>subjects · sessions · samples<br/>Hive-partitioned Parquet)]
    aevox[(data/aevox/v1<br/>phase 2, same schema)] -.-> parquet

    subgraph feat["Feature contract  (bioml.eval.features)"]
        fb[build_features<br/>27 wearable-only features · 8 groups<br/>causal rolling windows]
    end
    parquet --> fb

    subgraph model["Model  (bioml.train)"]
        lgb[3 × LightGBM<br/>RER · VO2 · P(above LT2)]
        meta[metadata.json<br/>feature contract · params · CV metrics · git sha]
    end
    fb --> lgb --> meta

    subgraph evalr["Evaluation  (bioml.eval.report)"]
        cv[subject-wise 5-fold CV<br/>2 baselines on identical folds<br/>calibration · SHAP · ablations]
    end
    fb --> cv

    subgraph infer["Inference  (bioml.inference)"]
        gate[Frayn gate<br/>rer &lt; 0.99 ∧ P(LT2) &lt; 0.5<br/>→ fat / CHO / kcal or 'buffering_dominated']
    end
    lgb --> gate

    subgraph serve["Serving  (bioml.serve)"]
        api[FastAPI /predict<br/>request → 3 tables → build_features]
        demo[Streamlit demo]
    end
    gate --> api
    gate --> demo
    api -. same code path .- fb
```

The dotted edge is the property the whole system is organised around: **the API and the trainer call the same `build_features`**, so train/serve skew is a tested invariant (`tests/test_serve.py::test_serving_features_match_training`) rather than a hope.

---

## 3. Repository map

```
src/bioml/
  schemas.py            Pydantic Subject / Session / Sample: the data contract
  config.py             research/spec.yaml loader + per-stratum/per-sex resolution
  frayn.py              Frayn 1983 stoichiometry + validity envelope
  generator/
    subjects.py         subject priors: bivariate (LT1,LT2) copula, lognormal MFO, Tanaka HRmax
    sessions.py         ramp CPET protocol; diet state, devices, ambient
    physiology.py       per-second engine: VO2 kinetics, 4-segment RER, glycogen, HR, Frayn
    sensors.py          PPG dropout/noise/cadence-lock, power-meter bias, CGM lag + MARD
    run.py              CLI: subjects → sessions → samples → Hive-partitioned Parquet
  eval/
    loaders.py          Dataset(subjects, sessions, samples: LazyFrame)
    features.py         THE feature contract: groups, causal rolling features, parity entry points
    splits.py           subject-wise k-fold (the only legitimate split here)
    metrics.py          MAE/bias by intensity, AUC, Brier, reliability/ECE, substrate-gate metrics
    report.py           `bioml evaluate`: baselines, calibration, SHAP, ablations → metrics.json + REPORT.md
  baselines/
    hr_zone.py          5-zone coaching heuristic (what a watch does today)
    population_curve.py bin-mean RER/VO2/P(LT2) vs %HRmax (a competent sports-science baseline)
  train.py              LightGBM multitask, TrainedModel artifact + metadata, generic CV loop
  inference.py          the one place the Frayn gate lives; session summary (Fatmax HR)
  intake.py             impute ffm / resting HR / training status from spec priors, with a log
  serve/
    schemas.py          request/response Pydantic models (bounds mirror the data schema)
    pipeline.py         request → 1 Hz grid → three tables → build_features → predictions
    app.py              FastAPI app factory, /health /model/info /predict, uvicorn entry
  demo/app.py           Streamlit: truth vs prediction for any synthetic subject
tests/                  136 tests (section 11)
research/               priors, sensor validation, design decisions, schema, spec.yaml
reports/v2/             metrics.json + REPORT.md produced by `make report`
```

---

## 4. Data layer

### 4.1 Three tables, one contract

| Table | Grain | Contents | Who fills it in phase 2 |
|---|---|---|---|
| `subjects` | one row per person | demographics, intake context, `*_true` lab parameters (VO2max, LT1/LT2, Fatmax, MFO, GE, tau, glycogen), the **wearable's** VO2max estimate | intake form + CPET report |
| `sessions` | one row per CPET | protocol, duration, latent diet/glycogen state (`*_true`, synthetic-only), ambient, device models | CPET metadata + user's device list |
| `samples` | one row per second | lab truth (`power_lab_w`, `vo2_lab_mlmin`, `vco2_lab_mlmin`, `rer_lab`, `hr_lab_bpm`, `pct_vo2max_lab`), derived truth (`fat_ox_g_per_min`, `cho_ox_g_per_min`, `above_lt2_flag`), wearable observations (`hr_wearable_bpm`, `power_wearable_w`, `cadence_rpm`, `cgm_glucose_mg_dl`) | metabolic cart (resampled to 1 Hz) + wearable export |

Design rules that matter:

- **Truth and observation are co-located but suffix-separated.** `*_true` / `*_lab` columns are labels and evaluation references only. `tests/test_eval.py::test_build_training_data_has_no_truth_leakage` fails if any such column reaches the feature matrix.
- **Nullable observations are the representation of missingness.** A PPG dropout is a null second. A rider without a power meter has a null power column for the whole session. The model trains on the real device-mix distribution rather than on an idealised one.
- **Derived truth is null where the chemistry is invalid.** `fat_ox_g_per_min` is null above LT2 rather than a biased number near zero. This is what later lets the gate be evaluated as a classifier.
- **Latent state is simulated but hidden.** Diet state and glycogen drive RER in the generator, but the model never sees them, because real users will not log them. The consequence is an irreducible RER error floor (section 6.3).

### 4.2 Storage

Parquet, Hive-partitioned by `subject_id` (`samples/subject_id=<uuid>/<session_id>.parquet`), read lazily with Polars (`scan_parquet(..., hive_partitioning=True)`). Partitioning by subject means a subject-wise fold is a predicate pushdown, not a shuffle; per-session files mean large datasets stream. The directory name carries the schema version (`v1`), so a schema change is a new directory, not a migration of old experiments.

The headline dataset is 100 subjects, 75,830 one-second rows, 7 MB on disk, regenerated bit-for-bit (same physiology; UUIDs and timestamps differ) from `make data` with seed 42 in under 3 seconds. It is gitignored and regenerable, which is the right trade for a portfolio repository.

Realised device mix in that dataset: 24% of sessions have no wearable HR at all, 20% no power meter, 66% no CGM; 28% of all seconds have null HR once dropouts are included; 15.9% of seconds are above LT2.

---

## 5. Synthetic generator

The generator is the project's physiology encoded as code. Each stage is a pure function of a seeded `numpy.random.Generator`, so the whole dataset is reproducible from one integer.

### 5.1 Subject priors (`generator/subjects.py`)

Each subject is a draw from `research/spec.yaml`: age uniform 18 to 65; sex 50/50; training status with weights 30/45/20/5 (untrained → elite). Every physiological parameter is then stratum- (and where the literature supports it, sex-) specific:

- **VO2max** truncated normal per stratum × sex at age 25, then a sampled age decrement (0.45 ± 0.15 mL/kg/min per year) is applied.
- **HRmax** = Tanaka (208 − 0.7·age) + N(0, 10): the structural mean a wearable also assumes, with the residual the literature reports.
- **(LT1, LT2)** drawn jointly from a truncated bivariate normal with r = 0.60 and a hard LT2 − LT1 ≥ 8 percentage-point floor, by rejection sampling. This is decision D1: thresholds are heterogeneous (Iannetta cohort SDs of 6 to 8 pp) and correlated within a person.
- **Fatmax** truncated below LT1 − 2 pp so the RER curve's knees stay ordered.
- **MFO** (peak fat oxidation) lognormal, because the inter-individual range is wide and right-skewed; keto-habitual subjects get a 2.3× multiplier (Volek 2016).
- Gross efficiency, VO2 kinetics tau, initial muscle glycogen: stratum priors.
- **The wearable's VO2max estimate** = true + N(−1.0, 3.5), capturing that consumer estimates are biased and noisy. The model sees the estimate, never the truth.

### 5.2 Session protocol (`generator/sessions.py`)

v1 implements one protocol, the ramp CPET: 180 s warm-up at 50 W, a linear ramp whose rate is the stratum's preferred rate bounded so the ramp lasts 3 to 20 minutes and terminates near the power at VO2max, then 180 s recovery at 50 W. Diet state is drawn conditional on habitual diet; pre-session glycogen fraction U(0.85, 1.0); ambient temperature N(21, 2) °C; wearable, power meter, and CGM models drawn uniformly including "none".

### 5.3 Per-second physiology (`generator/physiology.py`)

For each second:

1. **VO2** relaxes toward steady state (`VO2_rest + 287·P/GE` mL/min) with first-order kinetics `alpha = 1 − exp(−1/tau)`, capped at 1.06 × VO2max.
2. **%VO2max** from VO2 relative to rest and max.
3. **RER** = 4-segment piecewise-linear curve through anchors at rest, Fatmax, LT1, LT2, and 100% (stratum-specific anchor values), plus a glycogen-depletion drift (−β·(1 − glycogen fraction)), plus a latent diet offset (fasted 0, fed +0.04, keto −0.06), plus breath noise. This is decision D2. The Fatmax knee is what makes derived fat oxidation bell-shaped and peak at Fatmax; the steepening above LT2 represents bicarbonate buffering, not substrate change.
4. **VCO2** = RER × VO2.
5. **HR** linear in VO2 between resting and maximal HR.
6. **above_lt2_flag** = RER ≥ 1.00 or %VO2max > 85.
7. **Fat and CHO oxidation** via Frayn, only where valid (RER in [0.71, 1.00] and not above LT2); null otherwise.
8. **Muscle glycogen** depletes biphasically (decision D3): phase-1 rate anchored at 65% VO2max and scaled by (x/65)^1.8, dropping to 45% of that rate after the knee at 50 to 55% depletion. Anchored to Coyle 1986's measured 0.86 mmol/kg/min at 71% VO2max, which a test checks.

Deliberately not modelled in v1 (listed in the module docstring): the VO2 slow component above LT2, cardiovascular drift at constant intensity, and an exogenous-CHO blood-glucose channel. These matter for long steady-state sessions, not for a 10-minute ramp, which is why they are deferred alongside the protocols that need them.

### 5.4 Sensor degradation (`generator/sensors.py`)

Lab truth becomes device observation through per-device noise models from `research/sensor_noise.md`:

- **PPG heart rate.** Device-specific bias; intensity-regime noise SD (wrist 2 to 7.5 bpm, upper-arm 1 to 3 bpm); a 7-second causal moving average (manufacturer smoothing); dropout bursts of 10 to 60 s at regime-specific rates (wrist up to 25% of seconds at high intensity, upper-arm an order of magnitude lower); and **cadence lock**, where the optical sensor locks onto the pedal-stroke harmonic when true HR is within 10 bpm of cadence.
- **Power meter.** Per-device scale bias (0.7 to 2% SD) plus proportional noise; `None` for riders without one.
- **CGM.** Diet-dependent baseline, 5-minute (Dexcom G7) or 1-minute (Libre 3) sampling, mean reversion, exercise drift, multiplicative noise scaled to the device's MARD, and last-observation-carried-forward to 1 Hz.

Validation studies behind each parameter are cited in the research notes (Gillinov 2017, Falter 2019, Hettiarachchi 2019 for PPG; Garg 2022 and Alva 2022 for CGM; Lillo-Bevia 2021 and Hartog 2024 for power).

### 5.5 Why synthetic first

Three reasons, in order of importance. (1) There is no public dataset pairing breath-by-breath CPET with consumer wearable streams, so the pipeline has to be built before the data arrives. (2) A generator forces every physiological assumption into the open with a citation, which is exactly the review a hiring physiologist or clinician would perform. (3) Synthetic truth makes the evaluation honest in ways real data cannot: the Frayn validity envelope is known exactly, so the inference gate can be scored as a classifier.

The cost is equally clear: the model has learned the generator. Section 13 covers how phase 2 tests whether that transfers.

---

## 6. Feature engineering (`eval/features.py`)

### 6.1 The contract

`FEATURE_COLS` is a tuple of 27 names organised in 8 groups. The model's saved `metadata.json` pins this list; `TrainedModel.predict` raises `FeatureContractError` if a frame's columns or order differ. Both the trainer and the API obtain their matrix from `build_features`.

| Group | Features | Rationale |
|---|---|---|
| `demographics` | age, sex, height, weight, fat-free mass, training-status ordinal | intake form; FFM is the right denominator for metabolic rates |
| `fitness_context` | wearable VO2max estimate, resting HR | what the watch already knows about the person |
| `hr` | raw PPG HR, forward-filled HR, seconds since last valid HR, **%HRR estimate** | %HRR = (HR − HR_rest)/(HRmax_Tanaka − HR_rest) is the physiologist's normalisation of intensity; the gap counter makes missingness explicit |
| `hr_dynamics` | 30 s mean and SD, 60 s mean, 30 s slope | HR kinetics track VO2 kinetics; slope distinguishes rising from steady |
| `power` | raw power, 30 s mean, 30 s delta, cadence, **HR per watt** | external work is the strongest VO2 predictor when present; HR/W is an efficiency proxy |
| `cgm` | CGM glucose, 5-minute delta | the only wearable signal carrying fueling-state information |
| `time` | seconds since start, warm-up flag, recovery flag | informative on a ramp; a protocol shortcut in general (see ablations) |
| `environment` | ambient temperature | affects HR at a given VO2 |

### 6.2 Two properties that are tested, not assumed

**Causality.** Every rolling or lag feature is computed within a session, sorted by time, over a trailing window: `rolling_mean(window, min_samples).over("session_id")` and `shift(k).over("session_id")`. `tests/test_features.py::test_rolling_features_are_causal` perturbs every row after second 400 and asserts features before 400 are unchanged. A model that could see the future would score beautifully in CV and fail on a live stream.

**Parity.** The API reshapes a request into the three canonical tables and calls `build_features`. `test_serving_features_match_training` builds the same session both ways and asserts the matrices agree to 1e-9. When this test was first run it caught a genuine 1-ulp difference in `pct_hrr_est` from evaluation order, which is why the assertion is tolerance-based rather than bitwise.

### 6.3 The latent-variable floor

RER in the generator is a deterministic curve of intensity **plus** a diet offset (fasted 0, fed +0.04, keto −0.06) **plus** glycogen drift. The model sees neither diet nor glycogen. With roughly half of mixed-diet subjects fed and half fasted, the irreducible RER error from diet alone is about 0.02, and that is a real property of the deployment setting, not an artefact. It is why the RER MAE barely moved when rolling features were added (0.0342 → 0.0346) while VO2 MAE fell 7% and LT2 AUC rose. The CGM is the one wearable that carries fueling information, which the `no_cgm` ablation quantifies.

---

## 7. Model (`train.py`)

### 7.1 Three heads, one feature matrix

| Head | Target | Objective | Why |
|---|---|---|---|
| RER | `rer_lab` (0.70 to 1.15) | Huber, δ = 0.02 | robust to breath noise near RER 1.0; δ is at the residual scale (~0.03), and the D5 experiment confirmed L2 does not help here |
| VO2 | `vo2_lab_mlmin` | squared error (L2) | Huber at δ = 50 mL/min clipped gradients and starved the tails; decision D5, section 8.5 |
| LT2 | `above_lt2_flag` | binary cross-entropy, positive weight 4 | positives are ~16% of seconds; the gate needs recall |

Shared hyperparameters: learning rate 0.05, 31 leaves, min 50 rows per leaf, feature fraction 0.9, bagging 0.8 every 5 rounds, 400 rounds. These are defaults with light hand-tuning; there is no hyperparameter search, deliberately, because at n = 100 subjects a search would overfit the fold structure.

### 7.2 Why gradient-boosted trees, and why three separate boosters

- **Data scale.** 100 subjects, 76k rows, 27 features. A sequence model would be learning the generator's smoothness rather than physiology, and would need the real data to justify itself.
- **Nulls as signal.** LightGBM routes missing values natively, which matches the nullable-observation representation. No imputation layer that could leak a device assumption.
- **Attribution for free.** `pred_contrib=True` gives exact TreeSHAP values with no extra dependency; the report uses them out-of-fold so attribution is not measured on training rows.
- **Separate boosters, not one multi-output model.** Each head gets its own objective and loss scale. "Multitask" here means a shared feature contract and joint evaluation, not shared parameters. The D4 argument for the auxiliary LT2 head is that it receives gradient from the regime boundary independently of the regression and can be calibrated and gated on its own.

### 7.3 The artifact

`TrainedModel.save` writes three LightGBM text models plus `metadata.json` containing the feature names (also bound inside each booster, so `booster.feature_name()` is meaningful), hyperparameters, training-set size, LT2 positive rate, the cross-validated metrics the model shipped with, the data path, bioml and LightGBM versions, timestamp, and git SHA. `TrainedModel.load` refuses a directory without metadata. The API's `/model/info` returns the file verbatim, so a deployed model is self-describing.

---

## 8. Evaluation protocol (`eval/report.py`)

### 8.1 Splits

Subject-wise k-fold only. Each subject's whole session is in exactly one validation fold. Any row-wise or time-wise split leaks subject identity through the demographic features and the model becomes a memoriser of 100 people. `physiology_stratified_split` exists for a held-out set that covers all four training strata.

### 8.2 Baselines on identical folds

Beating zero is not a result. Two baselines run through the same `run_cv` loop and folds:

- **HR-zone heuristic** (`baselines/hr_zone.py`): the five-zone coaching table a watch uses today. Tanaka HRmax, canonical RER and VO2 fraction per zone, LT2 if above 87% HRmax. Fits nothing.
- **Population curve** (`baselines/population_curve.py`): bin-mean RER, VO2 fraction, and P(LT2) as functions of %HRmax, fitted on the training folds. This is what a competent sports scientist would build with a spreadsheet, and the bar the model has to clear convincingly.

### 8.3 What is measured

- RER and VO2: MAE, RMSE, signed bias, and both by %VO2max bin (0-30, 30-50, 50-65, 65-80, 80-90, 90-110). Bias by bin is what exposed the high-intensity VO2 shrinkage (8.5).
- LT2 head: AUC, Brier, a 10-bin reliability table, and expected calibration error (ECE). A gate that is discriminative but mis-calibrated would systematically emit fat oxidation where it should abstain.
- **Substrate-regime metrics** (`metrics.evaluate_substrate`): fat and CHO oxidation MAE/bias over rows where both the truth envelope and the inference gate say Frayn is valid, plus precision, recall, and accuracy of the gate as a classifier of truth validity. This is the metric of the thing the product actually outputs.
- Attribution: gain importance from the final model and mean |SHAP| pooled over out-of-fold predictions, per head.
- **Ablations**: six feature-group drop configurations, each a full 5-fold CV, answering deployment questions directly: no power meter, no CGM, watch only, no session clock, no HR dynamics, HR-only minimal.

### 8.4 Results

<!-- BEGIN:headline -->
| Model | RER MAE | VO2 MAE (mL/min) | LT2 AUC | LT2 Brier | fat-ox MAE (g/min) |
|---|---|---|---|---|---|
| HR-zone coaching heuristic | 0.0498 | 472 | 0.701 | 0.1152 | 0.186 |
| Population-mean curve | 0.0418 | 313 | 0.883 | 0.0875 | 0.071 |
| **LightGBM multitask v2** | **0.0346** | **113** | **0.959** | **0.0678** | **0.069** |

Relative to the population curve: RER MAE -17%, VO2 MAE -64%, LT2 Brier -23%, fat-ox MAE -2%. Relative to the HR-zone heuristic: RER MAE -31%, VO2 MAE -76%.

*5-fold subject-wise CV, n = 100 synthetic subjects (75,830 one-second rows), seed 42, 400 boosting rounds, 27 wearable-only features. Generated by `make report` (2026-11-05, bioml 0.2.0).*
<!-- END:headline -->

Substrate emission, the product output:

<!-- BEGIN:substrate -->
| Model | fat-ox MAE (g/min) | fat-ox bias | gate precision | gate recall | gate accuracy |
|---|---|---|---|---|---|
| HR-zone heuristic | 0.186 | +0.169 | 0.899 | 0.971 | 0.884 |
| Population curve | 0.071 | +0.007 | 0.903 | 0.967 | 0.884 |
| **LightGBM multitask v2** | 0.069 | -0.015 | 0.961 | 0.932 | 0.911 |
<!-- END:substrate -->

Error by intensity for the LightGBM model:

<!-- BEGIN:intensity -->
| %VO2max bin | rows/fold | RER MAE | RER bias | VO2 MAE | VO2 bias |
|---|---|---|---|---|---|
| 0-30% | 4,452 | 0.0240 | +0.0042 | 64 | +12 |
| 30-50% | 3,826 | 0.0339 | +0.0130 | 93 | +10 |
| 50-65% | 2,426 | 0.0404 | +0.0074 | 120 | +1 |
| 65-80% | 2,246 | 0.0402 | +0.0032 | 152 | -16 |
| 80-90% | 1,336 | 0.0398 | -0.0025 | 192 | -41 |
| 90-110% | 878 | 0.0473 | -0.0279 | 223 | -95 |
<!-- END:intensity -->

LT2 head calibration:

<!-- BEGIN:calibration -->
ECE = 0.0544

| P(LT2) bin | n | mean predicted | observed rate |
|---|---|---|---|
| 0.0-0.1 | 60,148 | 0.003 | 0.031 |
| 0.1-0.2 | 1,134 | 0.143 | 0.261 |
| 0.2-0.3 | 772 | 0.249 | 0.301 |
| 0.3-0.4 | 652 | 0.351 | 0.322 |
| 0.4-0.5 | 707 | 0.450 | 0.380 |
| 0.5-0.6 | 602 | 0.550 | 0.463 |
| 0.6-0.7 | 765 | 0.650 | 0.559 |
| 0.7-0.8 | 888 | 0.751 | 0.583 |
| 0.8-0.9 | 1,136 | 0.854 | 0.593 |
| 0.9-1.0 | 9,026 | 0.983 | 0.804 |
<!-- END:calibration -->

What each head relies on:

<!-- BEGIN:shap -->
| Head | Top features by out-of-fold mean abs SHAP |
|---|---|
| RER | `pct_hrr_est` (hr), `t_s` (time), `power_roll30_mean` (power), `vo2max_estimated_wearable_mlkgmin` (fitness_context), `is_recovery` (time), `hr_ffill_bpm` (hr) |
| VO2 | `power_roll30_mean` (power), `t_s` (time), `is_recovery` (time), `weight_kg` (demographics), `pct_hrr_est` (hr), `power_wearable_w` (power) |
| P(above LT2) | `pct_hrr_est` (hr), `t_s` (time), `hr_roll60_mean` (hr_dynamics), `power_roll30_mean` (power), `hr_roll30_mean` (hr_dynamics), `vo2max_estimated_wearable_mlkgmin` (fitness_context) |
<!-- END:shap -->

Feature-group ablations:

<!-- BEGIN:ablations -->
| Config | Dropped feature groups | #feat | RER MAE | Δ | VO2 MAE | Δ | LT2 AUC | fat-ox MAE |
|---|---|---|---|---|---|---|---|---|
| full | – | 27 | 0.0346 | – | 113 | – | 0.959 | 0.069 |
| no_time | time | 24 | 0.0378 | +9% | 154 | +36% | 0.947 | 0.072 |
| no_hr_dynamics | hr_dynamics | 23 | 0.0346 | -0% | 113 | +1% | 0.961 | 0.069 |
| no_power | power | 22 | 0.0347 | +0% | 130 | +16% | 0.954 | 0.070 |
| no_cgm | cgm | 25 | 0.0364 | +5% | 113 | +0% | 0.961 | 0.075 |
| watch_only | power, cgm | 20 | 0.0365 | +6% | 132 | +17% | 0.955 | 0.074 |
| hr_only_minimal | power, cgm, time, hr_dynamics, environment | 12 | 0.0449 | +30% | 260 | +131% | 0.894 | 0.083 |
<!-- END:ablations -->

### 8.5 Reading the results honestly

- **RER sits near its latent-variable floor** (section 6.3). The baselines are not far behind on RER (0.0418 for the population curve vs 0.0346) because the dominant unexplained variance is diet state, which none of them can see. The `no_cgm` ablation costs 5% RER MAE: the CGM is the one signal that carries fueling information, exactly as the physiology predicts.
- **Fat-oxidation MAE in the valid regime is only 2% better than the population curve** (0.069 vs 0.071 g/min), and that is the honest consequence of the RER floor: in the moderate-intensity range where Frayn applies, fat oxidation is a near-linear function of RER, so the two models' RER errors carry through. What the model does win there is the gate itself (precision 0.96 vs 0.90, accuracy 0.91 vs 0.88) and carbohydrate oxidation (MAE 0.205 vs 0.393 g/min), because CHO oxidation also depends on VO2, where the model is far better.
- **VO2 is where the model separates** (113 vs 313 mL/min, 64% lower), because power and the session clock are genuinely informative and the baselines use neither. SHAP agrees: the VO2 head is 62% power-driven by gain share, with the clock second.
- **The VO2 bias was a loss-function bug, and it is worth telling as a story.** With the original Huber objective (δ = 50 mL/min) the model under-predicted VO2 by 103 mL/min overall and by 592 mL/min in the 90-110% VO2max bin, while being unbiased below 50%. The pattern looked like tail shrinkage from noisy inputs. The actual mechanism: Huber's gradient is clipped at ±δ, so a residual of 600 mL/min contributes the same gradient as one of 50, and 400 rounds at learning rate 0.05 cannot move the extreme predictions far enough. Switching the VO2 head to squared error (decision D5) took overall bias to −5 mL/min, the top-bin bias to under 100, and cut VO2 MAE from 181 to 113 mL/min, with no change to the other heads. The spec had said δ = 15 and the code said δ = 50; both were wrong for a target whose residuals are hundreds of mL/min. The lesson that generalises: set a robust-loss δ relative to the residual scale you expect, and always look at bias by bin, because aggregate MAE hid this completely.
- **The ablations are the deployment story.** A watch-only customer (no power meter, no CGM) gets VO2 MAE 132 instead of 113 (+17%) and RER MAE 0.0365 (+6%), still far ahead of either baseline. Dropping the session clock costs +36% on VO2 and +9% on RER: that is the size of the ramp-protocol shortcut, and the first thing protocol diversity in the generator has to fix. Dropping HR-dynamics features changes nothing on a ramp, where the clock and power already determine intensity; they are in the contract for the steady-state and interval protocols to come, and the experiment to run next is `no_time` with and without them.
- **The LT2 head is discriminative but over-confident in the middle.** ECE 0.054, with predicted 0.75 against observed 0.58 in the 0.7-0.8 bin. The gate threshold of 0.5 still gives precision 0.96, so the product output is safe; an isotonic layer between the booster and the gate is the fix if real-data calibration looks the same.

---

## 9. Inference and the Frayn gate (`inference.py`)

Decision D4 in code. `emit_substrate(rer_pred, vo2_pred, lt2_prob)`:

```
valid  = (rer_pred < 0.99) AND (lt2_prob < 0.5)       # SubstratePolicy, both configurable
vco2   = rer_pred × vo2_pred
fat    = max(0, 1.695·VO2 − 1.701·VCO2)                 # g/min, L/min STPD
cho    = max(0, 4.344·VCO2 − 3.061·VO2)                 # exercise coefficients
energy = 9.75·fat + 4.18·cho                            # kcal/min, glycogen
regime = "substrate_calculable" if valid else "buffering_dominated"
fat, cho, energy = null where not valid
```

Why two conditions rather than thresholding RER alone: above LT2, bicarbonate buffering of lactic acid adds non-metabolic CO2, so RER keeps rising for reasons unrelated to fuel. A predicted RER of 0.97 with a high P(LT2) is a model that is uncertain about the regime, and the right output is abstention. The LT2 head gets gradient from the boundary independently of the regression, so the two conditions are not redundant.

`summarize_session` turns per-second output into what a coach reads: percentage of time buffering-dominated, mean and peak fat oxidation, and the **HR and power at peak predicted fat oxidation**, a wearable-only estimate of the subject's Fatmax zone. The demo shows this next to the truth from the lab curve.

The gate exists in exactly one module and is consumed by the report, the demo, and the API, so the three can never disagree about when fat oxidation is trustworthy.

---

## 10. Serving (`serve/`)

### 10.1 Contract

`POST /predict` takes a subject context, optional session context, and a list of per-second wearable samples; returns per-second predictions with regime labels plus a session summary. `GET /health` and `GET /model/info` expose the loaded artifact's metadata. Pydantic bounds mirror the data schema, so a 400 bpm heart rate or a 5 mL/kg/min VO2max is a 422 at the door.

### 10.2 Pipeline (`serve/pipeline.py`)

1. **Resolve context** (`intake.resolve_subject`). Fat-free mass, resting HR, and training status are optional. Missing values are imputed from the same `spec.yaml` priors the generator uses (training status by nearest age-adjusted stratum VO2max mean; FFM from the generator's body-fat assumption; resting HR from the stratum prior), and every imputation is reported in the response. Customers do not arrive with clean intake forms, and silently defaulting is worse than refusing.
2. **Reindex to a 1 Hz grid** (`samples_to_grid`). Timestamps must be strictly increasing but need not be contiguous; missing seconds become null sensor rows, which is exactly how dropouts are represented in training, so rolling windows keep their meaning.
3. **Three tables → `build_features`**, with the model's own `feature_names` as the column list.
4. **`predict_session`**, the shared gate, and `summarize_session`.
5. Return one row per requested second (grid fill-ins dropped), with NaN mapped to JSON null.

### 10.3 Operational notes

- Stateless per-session scoring: each request carries its own history, which is what the rolling features need. A streaming deployment would keep a per-session ring buffer of the last 300 seconds (the longest lag feature) and call the same builder on each tick.
- Measured locally against the final v2 model (three 400-tree boosters): a 600-second session scores in 28 ms median (p90 29 ms) including JSON (de)serialisation, over a warm HTTP connection on a laptop.
- `Dockerfile` builds a slim image with the model directory mounted at `/models/v2` rather than baked in, so one image serves any model whose `metadata.json` matches the installed feature contract. (Docker was not available on the development machine, so the Dockerfile is written to the published image conventions but not yet built here.)
- Not yet present and listed deliberately: authentication, request logging with PHI handling, prediction/feature drift monitoring, and a model registry. Section 14.

---

## 11. Testing strategy

136 tests in about two minutes. They are organised by what could silently go wrong:

| Layer | What is pinned | Examples |
|---|---|---|
| Physiology anchors | the generator reproduces published numbers | Frayn pure-fat/pure-CHO values; RER curve hits its anchors exactly; fat oxidation peaks within 5 pp of Fatmax; glycogen rate at 71% VO2max matches Coyle 1986 |
| Distributions | priors come out as specified | stratum proportions, VO2max means by stratum × sex, LT1/LT2 correlation ≈ 0.60, LT2 − LT1 ≥ 8, Fatmax < LT1, lognormal MFO skew, keto MFO uplift |
| Sensors | degradation has the right shape | dropout rises with intensity, upper-arm beats wrist, power scale within tolerance, Libre samples more often than Dexcom |
| Leakage | the model cannot cheat | no `_true`/`_lab`/`diet_state` column in features; subject-wise folds are disjoint |
| **Causality** | no feature sees the future | perturb t ≥ 400, features at t < 400 unchanged |
| **Parity** | API features == trainer features | same session through both paths, 1e-9 tolerance |
| Artifact | the contract is enforced | metadata written and read back; wrong or reordered columns raise `FeatureContractError`; missing metadata fails loudly; save/load predictions identical |
| Inference | the gate does what D4 says | both heads must agree; Frayn by hand; nulls outside regime; Fatmax HR from summary |
| Intake | imputation follows the priors | stratum means map to strata; age adjustment; every imputation logged |
| API | the service works end to end | health, model info, predict, imputation, gappy timestamps, five 422 cases |
| Model quality | the model beats the baseline | LightGBM beats the population curve on RER MAE in a 30-subject smoke run |
| CI | the CLIs run | GitHub Actions: ruff, ruff format, mypy strict, pytest, then generate → train → evaluate on a tiny dataset |

Static checks: `ruff` (pyflakes, pycodestyle, isort, pep8-naming, pyupgrade, bugbear, return, simplify) and `mypy --strict` on `src/`, both clean.

---

## 12. Reproducibility and operations

```bash
make install   # uv sync with ml, serve, dev extras (Python pinned to 3.12 by .python-version)
make data      # 100 subjects, seed 42 -> data/synthetic/v1      (~3 s)
make train     # 5-fold CV + final model with metadata -> models/v2
make report    # baselines, calibration, SHAP, 6 ablations -> reports/v2  (~7 min)
make serve     # FastAPI on :8000
make demo      # Streamlit
make check     # lint + typecheck + tests
uv run python scripts/sync_docs.py   # push report tables into README and this document
```

Provenance travels with every artifact: `metadata.json` carries the git SHA, versions, and CV metrics; `metrics.json` carries data path, seed, fold count, feature groups, and wall time. The dataset regenerates from seed; the model regenerates from the dataset; the docs regenerate from the report.

---

## 13. Phase 2: real CPET data

The schema was negotiated against the Aevox use case before any code was written (`research/schema_design.md`, "Aevox alignment"): breath-by-breath cart export, users bring their own devices, lactate optional, no reliable nutrition logging, one session per subject.

What changes:

- A loader (`data/aevox/v1/` producer) that parses the cart's breath-by-breath export, bins breaths to 1 Hz, and joins the wearable export on time with a ±1 to 2 s tolerance (device clocks drift). Threshold detection (LT1/LT2 from ventilatory equivalents) fills the `*_true` subject fields.
- `diet_state_true`, glycogen, and tau become null. Nothing downstream reads them.

What does not change: tables, feature builder, model code, evaluation harness, API.

What has to be validated before trusting anything:

1. **Sim-to-real gap on the generator's own terms.** Fit the population curve on real data; compare the real RER-vs-%HRmax curve with the synthetic one by stratum. Disagreement here is a prior that needs re-estimating (the "open items" list in `design_decisions.md` names the four most likely: RER anchors, the above-LT2 slope, the glycogen knee, the LT2 class weight).
2. **Transfer.** Train on synthetic, test on real, subject-wise. Then fine-tune or retrain on real with synthetic as augmentation, and report both.
3. **Sensor priors.** Measured dropout rates and PPG error by device on the real export replace the literature values.

---

## 14. Known limitations and roadmap

Honest list, roughly in the order an interviewer should raise them.

1. **One protocol.** Every session is a ramp, so `t_s` is nearly collinear with intensity. The `no_time` ablation measures the dependence; the fix is protocol diversity in the generator (steady-state and interval sessions, which also need the slow component and HR drift that v1 skips).
2. **The model has learned the generator.** Until phase 2 data exists, every metric is a statement about how well the pipeline recovers the generator's physiology through its noise model.
3. **Residual high-intensity VO2 under-prediction.** D5 removed most of it; what remains in the top bin is consistent with PPG dropout being highest there. Candidates if it matters for a customer: post-hoc isotonic calibration of the VO2 head, or an intensity-reweighted loss.
4. **No hyperparameter search**, by choice at this n. Revisit with real data.
5. **n = 100.** The generator produces 40 subjects per second; the number is a speed choice for CI and iteration, not a ceiling.
6. **Serving is a skeleton.** No auth, no structured request logging, no drift monitoring, no registry, no load test. Each is a known pattern; none is yet implemented.
7. **Calibration is measured, not enforced.** If ECE on real data is poor, a Platt or isotonic layer on the LT2 head goes between the booster and the gate.

---

## 15. Decision log

| Decision | Alternatives considered | Why this one | Where |
|---|---|---|---|
| Predict RER directly; gate Frayn downstream (D4) | regress fat-ox directly; censored regression on valid rows | fat-ox labels are biased toward zero above LT2 and would contaminate the valid range; censoring throws away 15-25% of data and leaves the boundary uncalibrated | `inference.py`, `design_decisions.md` |
| Auxiliary LT2 head rather than thresholding RER | threshold `rer_pred ≥ 1.0` | independent gradient at the regime boundary; calibratable; features (HR/VO2 mismatch) can signal buffering before RER reaches 1.0 | `train.py`, `inference.py` |
| 4-segment RER curve with a Fatmax knee (D2) | sigmoid centred at LT2; 3-segment | sigmoid puts the steepest slope at LT2 where buffering is still accelerating; 3-segment cannot produce the bell-shaped fat-ox curve | `generator/physiology.py` |
| Correlated, wide threshold priors (D1) | independent narrow priors | heterogeneity is biology (Iannetta n=1691), and LT1/LT2 co-vary within a person | `generator/subjects.py` |
| Biphasic-linear glycogen (D3) | exponential decay | Coyle 1986 serial biopsies show a knee, not a curve | `generator/physiology.py` |
| Squared-error VO2 head (D5) | Huber δ = 15 or 50 mL/min | Huber clipped the gradient at δ and under-predicted the top intensity bins by ~600 mL/min; L2 removed the bias and cut MAE 38% | `train.py`, `design_decisions.md` |
| Subject-wise CV only | row-wise, time-wise | demographics identify subjects; anything else is leakage | `eval/splits.py` |
| Two baselines on identical folds | report model alone | the HR-zone table is the incumbent; the population curve is the credible alternative | `baselines/`, `eval/report.py` |
| LightGBM, three boosters | multi-output NN; sequence model | data scale, native nulls, exact SHAP, per-head objectives | `train.py` |
| Named feature groups | flat list | ablations map to deployment questions (no power meter, watch only) | `eval/features.py` |
| Causal rolling features | bidirectional windows | the API scores live sessions; CV must reflect that | `eval/features.py`, `tests/test_features.py` |
| One `build_features` for train and serve | separate serving transform | train/serve skew is the classic production failure; make it a test | `serve/pipeline.py`, `tests/test_serve.py` |
| Impute intake fields from spec priors, log it | require all fields; silent defaults | customers' data is incomplete; silent defaults hide errors | `intake.py` |
| Reindex requests to a 1 Hz grid | require contiguous input | real exports have gaps; nulls are already the dropout representation | `serve/pipeline.py` |
| Model directory with metadata.json | pickle | self-describing, language-neutral text models, contract enforced on load and predict | `train.py` |
| Hive-partitioned Parquet by subject | one big file; database | subject folds are predicate pushdown; streams; versioned by directory | `generator/run.py`, `eval/loaders.py` |
| Synthetic data gitignored, regenerable | commit data | 7 MB regenerates in 3 s from a seed; provenance is the seed | `.gitignore`, `Makefile` |
| Docs tables generated from metrics.json | type numbers | numbers drift from code otherwise | `scripts/sync_docs.py` |

---

## 16. Glossary for the non-physiologist

- **CPET**: cardiopulmonary exercise test; incremental exercise with breath-by-breath gas analysis.
- **VO2 / VO2max**: oxygen consumption (mL/min, or mL/kg/min); its maximum is the standard fitness ceiling.
- **VCO2**: carbon dioxide production.
- **RER**: respiratory exchange ratio = VCO2/VO2. 0.71 means all fat, 1.00 all carbohydrate; above 1.00 the excess CO2 is from buffering, not fuel.
- **Frayn equations**: stoichiometry converting VO2 and VCO2 into grams per minute of fat and carbohydrate. Valid only when all expired CO2 is metabolic.
- **LT1 / LT2**: first and second lactate (or ventilatory) thresholds, bounding moderate, heavy, and severe intensity domains. LT2 ≈ the maximal lactate steady state.
- **Bicarbonate buffering**: above LT2, lactic acid is buffered by bicarbonate, releasing CO2 that inflates VCO2 and breaks Frayn.
- **Fatmax / MFO**: the intensity at which fat oxidation peaks, and the peak rate (g/min).
- **%HRR**: heart-rate reserve fraction, (HR − HR_rest)/(HR_max − HR_rest); the normalised intensity a watch can compute.
- **PPG**: photoplethysmography, the optical heart-rate sensing in wrist and arm wearables; prone to motion dropout and cadence lock.
- **CGM / MARD**: continuous glucose monitor; mean absolute relative difference, its standard accuracy metric (~8-13%).
- **Gross efficiency**: mechanical work out over metabolic energy in, ~19-24% in cycling; links power to VO2.
- **Huber loss**: squared error for small residuals, absolute for large; robust regression.
- **Subject-wise CV**: cross-validation where each person's data is entirely in one fold.
- **TreeSHAP**: exact Shapley attribution for tree ensembles; LightGBM computes it natively.
