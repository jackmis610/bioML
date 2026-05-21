# bioML

Predicting metabolic substrate use — **VO2, RER, and fat/carb oxidation rates** — from consumer wearable signals.

The project pairs a physiology-grounded synthetic data generator with a multitask gradient-boosted model. The synthetic phase ships standalone; the same schema and pipeline accept real CPET + wearable data via the Aevox partnership in phase 2.

## Headline results (5-fold subject-wise CV, n=100 synthetic subjects)

|                          | RER MAE | VO2 MAE (mL/min) | LT2 AUC | LT2 Brier |
|--------------------------|---------|------------------|---------|-----------|
| HR-zone coaching heuristic | 0.050   | 472              | 0.701   | 0.115     |
| Population-mean curve     | 0.042   | 313              | 0.883   | 0.088     |
| **LightGBM multitask v1** | **0.034** | **194**         | **0.953** | **0.070** |

LightGBM is **19–32% better than the strongest baseline** on RER MAE, **38–59% better** on VO2 MAE, and pushes LT2 detection to near-clinical AUC. Predictions use **wearable-only inputs** (PPG HR, optional power meter, optional CGM, demographics, watch-estimated VO2max) — lab signals are quarantined to training labels.

## What's encoded

Every choice in the generator and model is anchored to the literature. The `research/` directory is the source of truth:

- **`research/physiology_priors.md`** — quantitative metabolic-physiology priors with citations (Frayn 1983 substrate equations, Tanaka 2001 HRmax, Maunder 2018 MFO norms, Coyle 1986 glycogen depletion, Romijn 1993 substrate kinetics).
- **`research/sensor_noise.md`** — wearable validation studies (Gillinov 2017 PPG, Garg 2022 Dexcom G7, Lillo-Bevia 2021 power meters, etc.) → per-device noise / dropout / lag spec.
- **`research/design_decisions.md`** — four load-bearing calls: (D1) wider Iannetta-cohort threshold variability with bivariate copula, (D2) 4-segment piecewise-linear RER curve with a Fatmax knee that produces the bell-shaped fat-oxidation curve, (D3) Coyle-anchored biphasic-linear glycogen depletion, (D4) direct RER prediction with auxiliary LT2-flag multitask head + downstream Frayn gating.
- **`research/schema_design.md`** — canonical 3-table schema (subjects/sessions/samples) designed for one-file Aevox swap.
- **`research/spec.yaml`** — operational parameter config consumed by the generator.

## Architecture

```
src/bioml/
  frayn.py            substrate-oxidation stoichiometry + validity guards
  schemas.py          Pydantic Subject / Session / Sample
  config.py           spec.yaml loader
  generator/
    subjects.py       per-subject parameter sampling (bivariate copula on LT1/LT2, lognormal MFO)
    sessions.py       ramp CPET protocol; diet state, devices, ambient
    physiology.py     VO2 kinetics + 4-segment RER + biphasic glycogen + HR + Frayn
    sensors.py        PPG HR with dropout/cadence lock; power meter; CGM lag
    run.py            CLI: subject -> session -> samples -> Hive-partitioned parquet
  eval/
    loaders.py        hive-partitioned dataset load
    splits.py         subject-wise k-fold + physiology-stratified split
    features.py       wearable-only feature builder (no truth leakage)
    metrics.py        MAE, bias, intensity-binned, AUC, Brier
  baselines/
    hr_zone.py        5-zone coaching heuristic
    population_curve.py  bin-mean fit of (RER, VO2_frac, P(LT2)) vs %HRmax
  train.py            LightGBM multitask (Huber on RER + VO2, BCE on LT2)
  demo/app.py         Streamlit visualization

tests/                99 tests covering Frayn anchors, threshold copula
                      correlation, RER curve shape, glycogen depletion vs Coyle,
                      Frayn validity gating, sensor dropout rates, model
                      persistence round-trip, baseline-beat smoke test
```

## Quick start

```bash
# Install (uv installs into .venv)
uv sync --extra ml --extra serve --extra dev

# Generate synthetic dataset (40 subj/sec on a laptop)
uv run python -m bioml generate --n-subjects 200 --out data/synthetic/v1

# Train LightGBM, evaluate, and save model
uv run python -m bioml train --data data/synthetic/v1 --k 5 --n-estimators 400 --save-model models/v1

# Launch interactive demo
uv run streamlit run src/bioml/demo/app.py

# Run the test suite
uv run pytest
```

## Status

**Phase 1 — synthetic-data scaffold (complete).** 99 tests passing. CLIs work end-to-end. Streamlit demo renders. The synthetic data is physiologically defensible: RER curve hits anchor values exactly, fat-oxidation peaks within ±5 pp of subject Fatmax with a bell shape matching Achten 2003 / Maunder 2018, glycogen depletion at 71% VO2max matches Coyle 1986, Frayn gating exact across the entire dataset.

**Phase 2 — Aevox real-CPET integration (in progress).** Schema is designed so the Aevox loader is a one-file swap: same subjects/sessions/samples shape, same downstream pipeline, same eval harness. Once real data is in hand, the model retrains in minutes.

## License

MIT.
