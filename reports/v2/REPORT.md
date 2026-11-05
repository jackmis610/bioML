# bioML evaluation report

Generated 2026-11-05T19:01:09Z from `data/synthetic/v1` (100 subjects, 75,830 rows). 5-fold subject-wise CV, seed 42, 400 boosting rounds, 27 features. bioml 0.2.0 @ `949fb80`. Wall time 410.1s.

## Headline (same folds for every row)

| Model | RER MAE | VO2 MAE (mL/min) | VO2 bias | LT2 AUC | LT2 Brier | LT2 ECE |
|---|---|---|---|---|---|---|
| HR-zone heuristic | 0.0498 ± 0.0031 | 472 ± 53 | +333 | 0.701 | 0.1152 | 0.1152 |
| Population curve | 0.0418 ± 0.0034 | 313 ± 33 | +7 | 0.883 | 0.0875 | 0.0349 |
| **LightGBM multitask** | 0.0346 ± 0.0026 | 113 ± 13 | -5 | 0.959 | 0.0678 | 0.0554 |

LightGBM vs population curve: RER MAE -17.2%, VO2 MAE -64.0%, LT2 Brier -22.5%. vs HR-zone: RER MAE -30.5%, VO2 MAE -76.2%.

## Substrate emission (what the product actually outputs)

Fat/CHO oxidation is computed with Frayn only where the gate (`rer_pred < 0.99 AND P[LT2] < 0.5`) says it is valid. Truth validity is the generator's own Frayn envelope. Errors are over rows where both agree.

| Model | fat-ox MAE (g/min) | fat-ox bias | CHO-ox MAE (g/min) | gate precision | gate recall | gate accuracy | rows scored |
|---|---|---|---|---|---|---|---|
| HR-zone heuristic | 0.186 | +0.169 | 0.416 | 0.899 | 0.971 | 0.884 | 61,960 |
| Population curve | 0.071 | +0.007 | 0.393 | 0.903 | 0.967 | 0.884 | 61,651 |
| **LightGBM multitask** | 0.069 | -0.015 | 0.205 | 0.961 | 0.932 | 0.911 | 59,428 |

## Error by intensity (LightGBM)

| %VO2max bin | n/fold | RER MAE | RER bias | VO2 MAE | VO2 bias |
|---|---|---|---|---|---|
| 0-30% | 4,452 | 0.0240 | +0.0042 | 64 | +12 |
| 30-50% | 3,826 | 0.0339 | +0.0130 | 93 | +10 |
| 50-65% | 2,426 | 0.0404 | +0.0074 | 120 | +1 |
| 65-80% | 2,246 | 0.0402 | +0.0032 | 152 | -16 |
| 80-90% | 1,336 | 0.0398 | -0.0025 | 192 | -41 |
| 90-110% | 878 | 0.0473 | -0.0279 | 223 | -95 |

## LT2 head calibration (LightGBM, pooled out-of-fold)

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

## Feature attribution

### RER head: top features by out-of-fold mean |SHAP|

| Feature | Group | mean abs SHAP | gain share |
|---|---|---|---|
| `pct_hrr_est` | hr | 0.02965 | 40.7% |
| `t_s` | time | 0.0089 | 5.7% |
| `power_roll30_mean` | power | 0.007101 | 5.1% |
| `vo2max_estimated_wearable_mlkgmin` | fitness_context | 0.005731 | 6.3% |
| `is_recovery` | time | 0.004771 | 1.8% |
| `hr_ffill_bpm` | hr | 0.004769 | 4.9% |
| `cgm_glucose_mg_dl` | cgm | 0.004547 | 3.7% |
| `weight_kg` | demographics | 0.004521 | 5.5% |
| `ambient_temp_c` | environment | 0.004369 | 3.3% |
| `hr_rest_bpm` | fitness_context | 0.003794 | 5.2% |

### VO2 head: top features by out-of-fold mean |SHAP|

| Feature | Group | mean abs SHAP | gain share |
|---|---|---|---|
| `power_roll30_mean` | power | 379.8 | 61.5% |
| `t_s` | time | 188.7 | 17.9% |
| `is_recovery` | time | 68.39 | 3.0% |
| `weight_kg` | demographics | 34.43 | 0.8% |
| `pct_hrr_est` | hr | 30.45 | 1.0% |
| `power_wearable_w` | power | 29.82 | 4.5% |
| `ffm_kg` | demographics | 21.76 | 0.4% |
| `vo2max_estimated_wearable_mlkgmin` | fitness_context | 21.05 | 1.1% |
| `cadence_rpm` | power | 20.44 | 5.9% |
| `power_delta_30s` | power | 18.92 | 0.6% |

### LT2 head: top features by out-of-fold mean |SHAP|

| Feature | Group | mean abs SHAP | gain share |
|---|---|---|---|
| `pct_hrr_est` | hr | 2.197 | 46.8% |
| `t_s` | time | 1.673 | 9.5% |
| `hr_roll60_mean` | hr_dynamics | 0.9733 | 0.8% |
| `power_roll30_mean` | power | 0.8929 | 5.2% |
| `hr_roll30_mean` | hr_dynamics | 0.8676 | 1.3% |
| `vo2max_estimated_wearable_mlkgmin` | fitness_context | 0.6534 | 5.9% |
| `weight_kg` | demographics | 0.4675 | 2.2% |
| `hr_ffill_bpm` | hr | 0.3592 | 5.3% |
| `power_delta_30s` | power | 0.3328 | 3.0% |
| `hr_wearable_bpm` | hr | 0.3069 | 1.1% |

## Feature-group ablations (full subject-wise CV per row)

| Config | dropped | #feat | RER MAE | Δ | VO2 MAE | Δ | LT2 AUC | fat-ox MAE |
|---|---|---|---|---|---|---|---|---|
| full | – | 27 | 0.0346 | – | 113 | – | 0.959 | 0.069 |
| no_time | time | 24 | 0.0378 | +9.3% | 154 | +36.4% | 0.947 | 0.072 |
| no_hr_dynamics | hr_dynamics | 23 | 0.0346 | -0.0% | 113 | +0.7% | 0.961 | 0.069 |
| no_power | power | 22 | 0.0347 | +0.3% | 130 | +15.5% | 0.954 | 0.070 |
| no_cgm | cgm | 25 | 0.0364 | +5.2% | 113 | +0.1% | 0.961 | 0.075 |
| watch_only | power, cgm | 20 | 0.0365 | +5.6% | 132 | +17.0% | 0.955 | 0.074 |
| hr_only_minimal | power, cgm, time, hr_dynamics, environment | 12 | 0.0449 | +29.8% | 260 | +131.1% | 0.894 | 0.083 |

## Reproduce

```bash
uv run python -m bioml generate --n-subjects 100 --seed 42 --out data/synthetic/v1
uv run python -m bioml evaluate --data data/synthetic/v1 --k 5 --seed 42 --n-estimators 400 --out reports/v2
```
