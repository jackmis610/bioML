# bioML Generator & Model — Design Decisions

This document records the four load-bearing choices resolved on 2026-05-21 from a focused physiology-first-principles research pass, plus D5 (2026-11-05), a modeling decision revised by evidence from the evaluation harness. The detailed evidence is in [physiology_priors.md](physiology_priors.md) and the auxiliary research transcript; this file is the authoritative version of the decisions.

---

## D1. Threshold heterogeneity is wider than the initial prior; LT1 and LT2 are correlated

**Decision.** Replace the narrow LT1/LT2 SDs in §1.4 of the original physiology spec with the wider Iannetta-cohort priors below. Sample (LT1, LT2) jointly from a truncated bivariate Gaussian with Pearson r = 0.60. Enforce LT2 − LT1 ≥ 8 pp by rejection sampling.

| Stratum | LT1 mean ± SD (%VO2max) | LT2 mean ± SD (%VO2max) |
|---|---|---|
| Untrained / sedentary | 52 ± 8 | 81 ± 7 |
| Recreational          | 56 ± 7 | 85 ± 6 |
| Well-trained          | 62 ± 6 | 87 ± 5 |
| Elite endurance       | 68 ± 5 | 89 ± 4 |

**Why.** Iannetta cycle-ergometer cohort (n=1691, PMC12173951) reports SDs of 6–8 pp even with a single rigorous detection method — the heterogeneity is genuine biology, not method noise. VO2max correlates only weakly with %VO2max-at-threshold (r ≤ 0.115). Within-subject VT1↔VT2 partial correlation is r ≈ 0.55–0.65 on a %VO2max basis; LT2 − LT1 is bounded from below by physiological architecture (no recorded collapse < 8 pp).

**Citations.** Iannetta cohort PMC12173951; Pallarés 2016 *PLOS ONE* (well-trained cyclists, MLSS 73 ± 6 %); Faude 2009 *Sports Med* threshold review; Lucia 2000 *MSSE* (elite cyclists 88 ± 4–6 %).

---

## D2. RER vs %VO2max is 4-segment piecewise-linear, anchored at Fatmax/LT1/LT2 knees

**Decision (revised 2026-05-21).** Use a 4-segment piecewise-linear curve with anchors at (rest, Fatmax, LT1, LT2, max). The 3-segment form was upgraded after spot-checking the synthetic generator: the linear segment 0→LT1 can't reproduce a bell-shaped fat oxidation peak at Fatmax — fat_ox keeps rising monotonically up to LT1. Real data (Achten 2003 Fig 1, Maunder 2018) shows a clear bell shape peaking at Fatmax. Adding Fatmax as a knee resolves this:

| Stratum | RER_rest | RER at Fatmax | RER at LT1 | RER at LT2 | RER at 100% |
|---|---|---|---|---|---|
| Untrained    | 0.82 | 0.86 | 0.89 | 1.00 | 1.10 |
| Recreational | 0.82 | 0.87 | 0.90 | 1.00 | 1.08 |
| Well-trained | 0.82 | 0.88 | 0.91 | 1.00 | 1.06 |
| Elite        | 0.82 | 0.89 | 0.92 | 1.00 | 1.05 |

Anchor values were back-solved from the MFO priors: at Fatmax, derived fat_ox = VO2_at_Fatmax × (1.695 − 1.701 × RER_Fatmax) should equal MFO. For a typical well-trained male (VO2max 60 mL/kg/min, 75 kg, Fatmax 56%, MFO 0.53 g/min): RER_Fatmax ≈ (1.695 − 0.53/2.52) / 1.701 = 0.877 ≈ 0.88. Verified consistent across all four strata.

Subject sampler now enforces `Fatmax ≤ LT1 − 2 pp` so the 4-segment ordering is always valid.

Slopes derive automatically from (anchor, knee position) pairs sampled per subject:
- `s1 = (RER_LT1 − RER_rest) / LT1`
- `s2 = (RER_LT2 − RER_LT1) / (LT2 − LT1)`
- `s3 = (RER_max − RER_LT2) / (100 − LT2)`

Then RER(x) is the piecewise-linear curve through (0, RER_rest), (LT1, RER_LT1), (LT2, RER_LT2), (100, RER_max), with each knee softened by `softplus((x − knee) / Δ) · Δ`.

Add breath-by-breath noise σ ≈ 0.015 RER units per 30-s window (Bergman-Brooks day-to-day reliability).

**Why three segments.** The physiology is genuinely three regimes:
- **Below LT1**: shallow rise; substrate competition (FFA vs glucose) at submax mitochondrial drive.
- **LT1 → LT2**: steeper, near-linear; lactate accumulates and glycogenolysis accelerates.
- **Above LT2**: sharpest rise; **bicarbonate buffering** of lactic acid generates non-metabolic CO2 — this is the dominant driver of RER rise here, not substrate change.

A sigmoid centered at LT2 puts the steepest slope at LT2 itself; in real data the slope keeps climbing *above* LT2 because buffering accumulates with time-above-LT2. The 3-segment form respects this.

**Why anchor RER values, not slopes.** Parameterizing by slopes (per the agent's first table) overshot the canonical Achten/Bergman submax RER anchors. Anchor-point parameterization is self-consistent and reproducible across strata: trained subjects have lower RER throughout the curve (lower fasting + lower exercise RER at matched %VO2max), not just a lower LT2 knee. The agent's earlier slope table inverted this for some strata.

**Citations.** Achten & Jeukendrup 2003 *IJSM*; Bergman & Brooks 1999 *JAP*; San-Millán & Brooks 2018 *Sports Med* (48:467); Romijn 1993 *AJP*; Maunder 2018 *Front Physiol* (PMC5974542); Chenevière 2009 *MSSE* (SIN model fat-ox).

---

## D3. Muscle glycogen depletion is biphasic-linear; CHO feeding does NOT shift the muscle curve

**Decision.** Implement glycogen as a two-state piecewise-linear depletion with a knee at ~50–55% depletion:

```
G(t) = G0 − r1·t                              while G > G_knee
G(t) = G(t_knee) − r2·(t − t_knee)            while G ≤ G_knee
```

| Stratum | G0 (mmol/kg, range) | r1 at 65% VO2max (mmol/kg/min) | knee at fraction of G0 | r2/r1 |
|---|---|---|---|---|
| Untrained    | 95–115  | 0.45 ± 0.10 | 0.55 | 0.45 |
| Recreational | 110–135 | 0.60 ± 0.12 | 0.55 | 0.45 |
| Well-trained | 130–155 | 0.75 ± 0.10 | 0.50 | 0.45 |
| Elite        | 145–170 | 0.85 ± 0.10 | 0.50 | 0.45 |

Intensity scaling: `r1(x) = r1(65%) · (x / 65)^1.8`. Apply only for x ≤ 85% VO2max equivalent (above that, bout duration is short relative to the knee).

Carry log-normal noise σ ≈ 0.12 on both r1 and r2 (Coyle phase-2 CV ≈ 0.62 is genuinely high).

**CHO feeding (≥ 60 g/h):** does **not** alter the muscle-glycogen depletion curve. Model an *independent blood-glucose oxidation channel* (Coyle 1986 + Jensen 2025 meta-analysis confirm this is decisive). Liver glycogen is a separate state and *is* spared by CHO feeding — half-rate when CHO ≥ 60 g/h.

**Why biphasic-linear, not exponential.** Coyle 1986 *JAP* serial biopsies showed phase-1 utilization 51.5 ± 5.4 mmol GU·kg⁻¹·h⁻¹ for the first ~2 h, then phase-2 23.0 ± 14.3 — a ~55% slowdown at the knee, not a continuous exponential. The knee corresponds to fat oxidation up-regulating as muscle glycogen falls and FFA flux rises. Exponential decay misses this transition and would predict more total work-capacity than is observed.

**Why r1 values revised down from the agent's first proposal.** Coyle 1986 measured 0.86 mmol/kg/min at 71% VO2max in trained cyclists. The agent's "well-trained at 0.85 mmol/kg/min at 65% VO2max" scales via x^1.8 to ~1.0 at 71%, which exceeds Coyle. Re-anchoring r1(well-trained, 65%) at 0.75 gives r1(71%) ≈ 0.88 — matches Coyle.

**Citations.** Coyle 1986 *JAP* 61:165–172; Hawley & Leckey 2015 *Sports Med* (PMC4672006); Romijn 1993 *AJP*; Jensen 2025 *JAP* meta-analysis on CHO + muscle glycogen; Gonzalez 2015 *AJP-Endo* on liver vs muscle glycogen sparing.

---

## D4. Predict RER directly with a multitask LT2-flag head; gate Frayn at inference

**Decision.** Adopt **direct RER prediction (option C) with a multitask binary classifier head for the above-LT2 regime**.

- **Primary target:** RER (continuous, range 0.70–1.15). Trained on the full intensity range.
- **Auxiliary target (shared encoder):** `above_LT2_flag` ∈ {0, 1}, defined as RER ≥ 1.0 OR VO2 > 85% VO2max steady-state.
- **Losses:** Huber on RER with δ = 0.02 (robust to breath-by-breath noise and to ceiling effects in the buffering regime). BCE on the LT2 flag with positive-class weight ≈ 4 (the LT2-positive segments are a minority of session time).
- **Downstream fat_ox emission:** Apply Frayn only when `predicted RER < 0.99` **AND** `LT2_flag probability < 0.5`. Otherwise return `fat_ox = NA, regime = "buffering-dominated"`.
- **Evaluation:**
  - Valid range (RER < 0.99): MAE and bias of fat_ox (g/min); Bland-Altman; per-intensity-bin error; calibration plot of predicted RER vs true RER.
  - Invalid range: AUC and calibration of LT2 flag classifier; RER MAE only — no fat_ox metric.
  - Subject-wise k-fold CV with a physiology-stratified held-out test split (subjects spanning all four training strata).

**Why option C over A or B.**
- (A) Single-target fat_ox regression: above LT2, training labels are *systematically biased toward zero* by the Frayn breakdown. The model learns a spurious "fat_ox crashes near max" pattern that contaminates the LT1–LT2 range too. Reject.
- (B) Censored regression: train only on RER < 1.0 segments. Discards 15–25% of session data; classifier near the boundary is hard to calibrate without joint training.
- (C) Direct RER: the model targets a *smooth, well-defined physiological signal* on the entire intensity range. Frayn is applied downstream where it's actually valid. The wearable-ML literature consistently chooses this pattern — predict the underlying continuous variable, apply transformations downstream.

**Why a multitask LT2 head rather than thresholding predicted RER.** Mechanically the flag is mostly `RER_pred ≥ 1.0`, so it sounds redundant. But the auxiliary head (a) gets independent gradient signal from segments where the regression is uncertain, (b) can pick up features that signal the buffering regime *before* RER hits 1.0 (e.g., HR/VO2-estimate mismatch, ventilation-related cadence patterns), and (c) provides a calibrated safety gate at inference that's separately interpretable from RER prediction error.

**Citations.** Frayn 1983 *JAP*; Romijn 1993 *AJP* (explicit Frayn ceiling at RER 1.0); Hawley & Leckey 2015 (substrate-modeling Frayn ceiling guidance); Chenevière 2009 *MSSE* (SIN fat-ox model); PMC9585001 substrate-modeling analysis; PMC11720257 dynamic-prediction wearable-ML pattern; PMC11617143 endurance-athlete ML mini-review.

---

## D5. The VO2 head uses squared error; Huber clipped the gradient and starved the tails

**Decision (2026-11-05).** Train the VO2 booster with the plain L2 objective. Keep Huber (δ = 0.02) on the RER booster.

**What was observed.** With Huber δ = 50 mL/min on VO2 (the v1 setting; the schema doc had said δ = 15), 5-fold subject-wise CV gave VO2 MAE 181 mL/min and an overall bias of −103 mL/min that was invisible in aggregate but stark by intensity bin: +20 mL/min below 30% VO2max, −198 at 65–80%, −402 at 80–90%, −592 at 90–110%. The shape looked like regression toward the mean from noisy inputs where PPG dropout is highest.

**Mechanism.** Huber's gradient is clipped at ±δ. A residual of 600 mL/min pushes the ensemble exactly as hard as one of 50, so with learning rate 0.05 and 400 rounds the extreme predictions cannot travel far enough. The δ was set at the wrong scale for a target whose residuals are hundreds of mL/min.

**Experiment (same folds, VO2 head only):**

| VO2 objective | MAE (mL/min) | bias | bias in 90–110% bin |
|---|---|---|---|
| Huber δ = 50 (v1) | 181 | −103 | −592 |
| Huber δ = 300 | 118 | −10 | −110 |
| **L2** | **113** | **−5** | **−95** |
| L2, 63 leaves, min 20/leaf | 112 | −2 | −79 |

The larger-capacity variant buys 1 mL/min; not worth the variance at n = 100.

**The control.** The same experiment on the RER head changed nothing (Huber δ = 0.02: MAE 0.0346; L2: 0.0357; Huber δ = 0.10: 0.0355; per-bin biases within 0.003 of each other). RER residuals are ~0.03, so δ = 0.02 is at scale and does not starve the gradient; the remaining RER tail bias (−0.028 in the top bin) is the latent diet/anchor floor, not the loss. Huber stays, because it is more robust to the per-second breath noise near RER 1.0.

**Lessons that generalise.** Set a robust-loss δ relative to the residual scale you expect, not to the units that feel small. Always report signed bias by bin; aggregate MAE hid a 600 mL/min systematic error. And run the same experiment on the head where you expect no effect, to confirm you have found a mechanism and not a coincidence.

---

## Open items deferred to data-driven tuning

Once real Aevox CPET + wearable data is available, the following parameters should be re-estimated rather than carried forward unchanged:
1. The exact RER anchor values per stratum — current values are population priors, real data will give subject-specific shifts.
2. The s3 slope in the buffering regime — bicarbonate buffering varies substantially with training and is hard to anchor without test-protocol-specific data.
3. The knee fraction (0.50 vs 0.55 of G0) — Coyle's single study has limited statistical power on this exact number.
4. The LT2-flag positive-class weight — currently set heuristically at 4; should be tuned against observed class balance in real sessions.
