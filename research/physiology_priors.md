# Synthetic CPET Generator: Parameter Spec

Per-second cycling-ergometer CPET + wearable simulation for healthy adults (18-65, both sexes, untrained → elite endurance). All values are population priors for a generative model; treat means/SDs as Gaussian truncated to physiological bounds unless otherwise noted (lognormal noted where appropriate).

Confidence levels: **A** = meta-analysis or pooled normative dataset, **B** = consistent single-lab/multi-lab evidence, **C** = single study or textbook consensus with notable controversy.

---

## 1. Per-subject parameter priors

### 1.1 VO2max (mL·kg⁻¹·min⁻¹)

Population reference is the LowLands Fitness Registry (n≈4,800, cycle ergometer, ages 6-65) plus elite-cyclist sources. Decline with age in adults is ≈0.4-0.5 mL·kg⁻¹·min⁻¹·yr⁻¹.

| Stratum | Sex | Mean | SD | Notes / Citations | Conf |
|---|---|---|---|---|---|
| Untrained | M | 38 | 7 | LowLands (Loe 2014/Bongers 2021, BJSM); 30-45 typical adult range | A |
| Untrained | F | 32 | 6 | LowLands; ~18% lower than M at matched age | A |
| Recreational | M | 50 | 7 | ACSM Guidelines 11e percentile 70-85 | A |
| Recreational | F | 42 | 6 | ACSM | A |
| Well-trained endurance | M | 62 | 6 | Lucia et al. 2001 Sports Med; multiple cyclist cohorts | B |
| Well-trained endurance | F | 54 | 6 | Sandbakk & Holmberg 2017 IJSPP | B |
| Elite endurance | M | 78 | 5 (range 70-90) | Pogačar ~89, Froome 84.6, TdF 76-85 (Velora/Rouleur, Lucia 2001) | B |
| Elite endurance | F | 70 | 5 (range 62-80) | Sandbakk 2017; world-class female XC ski/road | B |

Apply age decrement: VO2max(age) = VO2max(25) − 0.45·(age−25); SD on decrement slope ≈0.15.

### 1.2 HRmax

**Use Tanaka** (208 − 0.7·age) as the structural mean; it outperforms Fox (220−age) above age 40 (Tanaka 2001 *JACC*; Robergs & Landwehr 2002 *J Exerc Physiol*). Residual SD ≈ **7-11 bpm**; individual prediction error can reach ±20 bpm. Nes equation (211 − 0.64·age, *Scand J Med Sci Sports* 2013) is slightly higher; treat as alternative.

```
HRmax_mean = 208 − 0.7·age
HRmax_sd   = 10  (Tanaka 2001; reanalyses ~7.2-11)
```
No reliable sex difference in HRmax across meta-analyses. **Confidence A.**

### 1.3 Resting HR

Lognormal-ish; truncated normal acceptable.

| Stratum | Mean (bpm) | SD | Citation | Conf |
|---|---|---|---|---|
| Untrained M | 70 | 10 | AHA / NHANES | A |
| Untrained F | 74 | 10 | NHANES (women ~4 bpm higher) | A |
| Recreational | 60 | 8 | Carter 2003 *Sports Med* | B |
| Well-trained | 50 | 7 | ibid; Master athletes 62.8±6.7 (D'Andrea 2015) | B |
| Elite endurance | 42 | 6 | Range 35-50; case reports to 28 | B |

### 1.4 LT1 / LT2 (ventilatory equivalents)

LT1 ≈ VT1 ≈ aerobic threshold; LT2 ≈ VT2 ≈ RCP ≈ MLSS (within ~5% VO2max).

| Stratum | %VO2max @ LT1 | %VO2max @ LT2 | %HRmax @ LT1 | %HRmax @ LT2 | Source | Conf |
|---|---|---|---|---|---|---|
| Untrained | 50 ± 6 | 70 ± 6 | 65 ± 6 | 82 ± 5 | Davis 1979; Bhambhani 1985 *MSSE*; Caiozzo 1982 | B |
| Recreational | 58 ± 6 | 77 ± 5 | 70 ± 5 | 86 ± 4 | ibid | B |
| Well-trained | 65 ± 5 | 83 ± 4 | 76 ± 4 | 90 ± 3 | Pallarés 2016 *PLOS ONE* (well-trained cyclists VT2 ≈83%) | B |
| Elite endurance | 72 ± 4 | 88 ± 3 | 80 ± 3 | 92 ± 3 | Lucia 2000 *JAP*; Coyle 1991 | B |

Note: VT2 ↔ MLSS agreement is ~r=0.85; expect ±5% VO2max noise between protocols.

### 1.5 Fatmax (%VO2max at peak fat oxidation)

From Maunder et al. 2018 *Front Physiol* normative tables (cycle, fasted):

| Stratum | Fatmax (%VO2max) | SD | Conf |
|---|---|---|---|
| Endurance-trained M | 56 | 8 | A |
| Recreational M | 51 | 8 | A |
| Recreational F | 50 | 10 | A |
| Untrained (any sex) | 48 | 10 | B (interpolated) |
| Elite endurance | 60 | 7 | B (San-Millán & Brooks 2018) |

### 1.6 Peak fat oxidation rate (MFO, g·min⁻¹)

Maunder 2018 pooled cycle data (fasted). **Very wide inter-individual range** — preserve this in the prior.

| Stratum | Mean | SD | 20-80%ile range | Conf |
|---|---|---|---|---|
| Endurance-trained, lean M (n=201) | 0.53 | 0.16 | 0.40-0.67 | A |
| Recreational, lean M (n=105) | 0.46 | 0.14 | 0.34-0.58 | A |
| Recreational, lean F (n=68) | 0.35 | 0.12 | 0.25-0.45 | A |
| Untrained (any sex) | 0.30 | 0.12 | 0.20-0.42 | B |
| Elite endurance M | 0.65 | 0.18 | up to 1.0+ (San-Millán 2018) | B |
| Elite endurance F | 0.55 | 0.15 | — | C |
| Keto-adapted ultra-endurance M | 1.54 | 0.18 | Volek 2016 *Metab Clin Exp* | B |

Achten & Jeukendrup 2003 *Med Sci Sport Exerc*: trained men 0.52 ± 0.15 g·min⁻¹ at 62.5 ± 9.8% VO2max.

### 1.7 %VO2max at crossover

Brooks & Mercier 1994 *JAP* — defined where CHO-derived kcal exceed fat-derived kcal. Original concept claims ~50% VO2max for untrained, shifted right with training. Significant controversy (Coggan 1997 *J Appl Physiol*; meta-analyses note crossover is essentially a re-expression of the RER vs intensity curve). Practically:

| Stratum | Crossover (%VO2max) | Range | Conf |
|---|---|---|---|
| Untrained | 45 | 35-55 | C |
| Recreational | 55 | 45-65 | C |
| Well-trained | 65 | 55-75 | C |
| Elite endurance | 70 | 60-80 | C |

Above ~85% VO2max, fat contribution is near-zero in non-keto subjects (Romijn 1993 *AJP*).

### 1.8 Cycling gross efficiency (GE)

Moenster-Pannekoek & van Ingen Schenau / Coyle 1992 / Mogensen 2006 / Hopker 2010 *MSSE*:

| Stratum | GE % (moderate, ~150-200 W or ~60% VO2max) | SD | Conf |
|---|---|---|---|
| Untrained | 19.0 | 1.5 | B |
| Recreational | 20.5 | 1.5 | B |
| Well-trained | 22.0 | 1.3 | B |
| Elite endurance | 23.5 | 1.5 (up to 25-26 in time-trial specialists) | B |

GE rises mildly with intensity (~0.3-0.5%-point from 50→80% VO2max in trained subjects, Moseley 2004), then declines above VT2. Treadmill: running economy is more variable; gross efficiency ≈ 19-22% but typically reported as running economy (mL·kg⁻¹·km⁻¹) instead.

### 1.9 Resting RER

| Condition | Mean RER | SD | Citation | Conf |
|---|---|---|---|---|
| Overnight fasted (12h), mixed-diet adult | 0.82 | 0.04 | Goedecke 2000 *AJP*; Galgani 2008 | A |
| 2-3 h post-mixed meal | 0.88 | 0.04 | ibid | A |
| Habitual high-fat / keto | 0.74 | 0.03 | Volek 2016 | B |
| Habitual high-CHO (>60% kcal) | 0.86 | 0.04 | Schrauwen 1997 | B |

---

## 2. Within-session dynamics

### 2.1 RER vs %VO2max (anchor points)

Standard form is a monotonic increasing sigmoid bounded ~[0.78, 1.10] during a ramp/incremental, with an inflection near LT2. Practical analytic form:

```
RER(%VO2max) = RER_rest + (RER_max − RER_rest) / (1 + exp(−k·(%VO2max − x0)))
```
where x0 ≈ LT2 (% VO2max for that subject), k ≈ 0.10-0.15 (steeper for trained), RER_max ≈ 1.05-1.15 at exhaustion (CO2 from bicarbonate buffering > metabolic CO2).

Anchor values during steady-state cycling, fasted (Achten 2003, Bergman & Brooks 1999 *JAP*, Romijn 1993):

| %VO2max | Untrained RER | Trained RER |
|---|---|---|
| 40 | 0.84 | 0.81 |
| 50 | 0.87 | 0.83 |
| 60 | 0.91 | 0.86 |
| 70 | 0.95 | 0.90 |
| 80 | 0.99 | 0.95 |
| 90 | 1.02 | 1.00 |

Add steady-state noise σ ≈ 0.015 RER units per 30-s window.

### 2.2 Substrate shift during prolonged exercise at ~60% VO2max

Romijn 1993 / Coyle 1986 / Watt 2002. At constant moderate intensity (60-65% VO2max), RER drifts down over time (fat oxidation rises as muscle glycogen depletes):

| Time | ΔRER from t=15 min | Note |
|---|---|---|
| 0-15 min | baseline (~0.91 untrained / 0.86 trained) | initial CHO-dominant |
| 60 min | −0.02 | early lipolytic activation |
| 90 min | −0.04 | |
| 120 min | −0.05 to −0.07 | glycogen ~50% depleted |
| 180 min | −0.07 to −0.10 | RER may reach 0.80; Coyle 1986: 0.85 → 0.80 pre-fatigue |

Tie to muscle glycogen with first-order kinetics: glycogen(t) = G0·exp(−t/τg) with τg ≈ 90-150 min at 65% VO2max (untrained shorter, trained longer due to higher glycogen reserves).

### 2.3 Carb intake (~60 g·h⁻¹) effect on RER

Cox 2010 *J Appl Physiol*; Smith 2010 *MSSE*; Jeukendrup 2014: 60 g·h⁻¹ exogenous CHO during 2-3 h cycling at 60-65% VO2max:

- RER ≈ **+0.03 to +0.05** higher vs. 0 g·h⁻¹ from ~45 min onward
- Suppresses the RER drift in §2.2 — drift to t=180 min reduced to ~−0.02
- Total fat ox suppressed by ~30-45%
- Plateau of exogenous CHO oxidation at ~1.0-1.1 g·min⁻¹ for single glucose source

### 2.4 Pre-exercise carb meal vs fasted

Wallis 2006 *MSSE*; Bennard & Doucet 2006:

- 2-3 h after CHO meal (~1-2 g·kg⁻¹): resting RER +0.04 to +0.06 vs fasted
- Exercise RER at matched 60% VO2max: +0.03 to +0.05 vs fasted; fat ox 30-50% lower
- Effect attenuates as intensity rises above LT2

### 2.5 VO2 kinetics τ (phase II)

Koppo 2004 *MSSE*; Caputo & Denadai 2004 *EJAP*:

| Stratum | τ moderate (s) | τ heavy (s) | Conf |
|---|---|---|---|
| Untrained | 35 ± 8 (range 25-50) | 50 ± 12 | B |
| Recreational | 25 ± 6 | 35 ± 10 | B |
| Well-trained | 18 ± 4 | 25 ± 8 | B |
| Elite endurance | 10 ± 3 | 15 ± 5 | B |

Mean response time ≈ τ + cardiodynamic delay (~15-20 s).

### 2.6 VO2 slow component

Magnitude above LT2 (Jones 2011 *MSSE*): trivial moderate (~10-50 mL/min), heavy ~100-200 mL/min, severe 400-700 mL/min, extreme 1000-1500 mL/min. Onset time ~80-120 s after exercise start. Smaller in trained subjects (~30-40% reduction).

### 2.7 HR drift at constant moderate power

Coyle & Gonzalez-Alonso 2001 *Exerc Sport Sci Rev*; Mognoni 1990:

- At 60-65% VO2max, temperate environment, normohydration: **HR rises ~3-5 bpm per hour** (~5-8% over 2 h)
- Heat (>30°C) or dehydration: **10-20 bpm·h⁻¹**
- Stroke volume falls proportionally; VO2 stays constant
- Trained subjects show less drift (~3 bpm·h⁻¹) than untrained (~6 bpm·h⁻¹)

Simulation form: HR(t) = HR_ss + α·t + ε, with α ~ N(0.07, 0.03) bpm·min⁻¹ at 65% VO2max temperate; multiply by ~3 in heat.

---

## 3. Stoichiometric / Frayn equations

Standard non-protein RQ approach (Frayn 1983 *JAP*; Péronnet & Massicotte 1991 *Can J Sport Sci*; Jeukendrup & Wallis 2005 *IJSM*):

```
# Resting / low intensity (glucose-dominant CHO):
fat_ox  (g/min) = 1.695·VO2 − 1.701·VCO2 − 1.77·N
CHO_ox (g/min) = 4.55·VCO2 − 3.21·VO2 − 2.87·N

# Moderate-high intensity exercise (muscle glycogen-dominant):
fat_ox  (g/min) = 1.695·VO2 − 1.701·VCO2 − 1.77·N
CHO_ox (g/min) = 4.344·VCO2 − 3.061·VO2 − 0.40·N
```
VO2, VCO2 in L/min STPD; N = urinary nitrogen excretion (g/min); typically set N=0 (protein contribution assumed <5% of energy; introduces ≤2% error). Energy: 1 g fat = 9.75 kcal, 1 g CHO = 4.07 kcal (glucose) or 4.18 kcal (glycogen).

**Valid RER range: 0.71-1.00**. At RER < 0.71 (rare except in fat-adapted), fat_ox calc still works but is dubious. **At RER ≥ 1.00**, fat_ox is mathematically zero or negative — clamp to 0 in code. Above LT2, bicarbonate buffering inflates VCO2 (non-metabolic CO2) and invalidates the equations (Jeukendrup & Wallis 2005). Don't compute substrate rates above ~85% VO2max steady-state without correction.

Protein correction is typically omitted in exercise; introduces ≤3% error in fat_ox at rest (Romijn 1993).

---

## 4. Training-status shifts (untrained → elite)

| Parameter | Direction | Magnitude (un→elite) | Source |
|---|---|---|---|
| VO2max | ↑ | +40 mL/kg/min (≈2× absolute) | LowLands; Lucia 2001 |
| MFO | ↑ | 0.30 → 0.65+ g/min (~2×) | Maunder 2018 |
| Fatmax (%VO2max) | ↑ | 48% → 60% | Nordby 2006; San-Millán 2018 |
| Crossover (%VO2max) | ↑ | 45% → 70% | Brooks 1994 |
| VO2 kinetics τ | ↓ | 35 s → 10 s | Koppo 2004 |
| HR drift | ↓ | 6 → 3 bpm·h⁻¹ | Coyle 2001 |
| GE | ↑ | 19% → 23-24% | Hopker 2010 |
| LT2 (%VO2max) | ↑ | 70% → 88% | Coyle 1991 |

---

## 5. Diet effects (keto-adapted vs high-carb)

Volek 2016 *Metab Clin Exp* (FASTER); Burke 2017 *J Physiol*; Phinney 1983:

- Keto-adapted MFO: **1.5 g·min⁻¹ avg, peaks 1.8** (vs 0.7-1.0 in HC athletes); 2-2.3× higher
- Fatmax shifted right: ~70% VO2max (vs ~55-60% HC)
- Submax exercise RER at 64% VO2max: **0.73-0.75** (keto) vs 0.86-0.89 (HC)
- Crossover delayed to ~80%+ VO2max
- Cost: impaired high-intensity CHO availability; equivocal/negative effect on race performance (Burke 2017)

---

## 6. Sex differences (brief)

- At matched %VO2max (moderate intensity), women oxidize **~5-10% more energy from fat** than men (Carter 2001 *JAP*; Venables 2005 *JAP*). Reflected in slightly higher Fatmax (~+3-5 %VO2max) and lower RER (~-0.02 to -0.03).
- Absolute MFO lower in women (~0.35 vs 0.46 g·min⁻¹) but **higher when normalized to fat-free mass**.
- Estrogen-mediated; cycle phase introduces ~0.01-0.02 RER variability (luteal vs follicular).
- No reliable sex difference in HRmax, LT1/LT2 %VO2max, VO2 kinetics, or HR drift.

---

## 7. Starter Parameter Table (YAML, drop-in)

```yaml
# Synthetic CPET subject sampler -- priors for healthy adults 18-65, cycling ergometer
# All values: (mean, sd) unless noted (range) or distribution specified
# Use Gaussian truncated to physiological bounds unless noted

age:
  distribution: uniform
  bounds: [18, 65]

sex_ratio_M: 0.5

training_status_weights:  # multinomial
  untrained:     0.30
  recreational:  0.45
  well_trained:  0.20
  elite:         0.05

# ---- VO2max (mL/kg/min, at age 25; apply age decrement = 0.45/yr after 25) ----
vo2max:
  untrained:     { M: [38, 7],   F: [32, 6] }
  recreational:  { M: [50, 7],   F: [42, 6] }
  well_trained:  { M: [62, 6],   F: [54, 6] }
  elite:         { M: [78, 5],   F: [70, 5] }
  age_decrement_per_year: [0.45, 0.15]   # mean, sd of subject-specific slope
  bounds: [20, 95]

# ---- HRmax (bpm) ----
hrmax:
  formula: "208 - 0.7*age"
  residual_sd: 10
  bounds: [150, 220]

# ---- Resting HR (bpm) ----
hr_rest:
  untrained:     { M: [70, 10],  F: [74, 10] }
  recreational:  { all: [60, 8] }
  well_trained:  { all: [50, 7] }
  elite:         { all: [42, 6] }
  bounds: [30, 100]

# ---- Thresholds (% VO2max) ----
lt1_pct_vo2max:
  untrained:     [50, 6]
  recreational:  [58, 6]
  well_trained:  [65, 5]
  elite:         [72, 4]

lt2_pct_vo2max:
  untrained:     [70, 6]
  recreational:  [77, 5]
  well_trained:  [83, 4]
  elite:         [88, 3]
  constraint: "lt2 > lt1 + 8"

# ---- Fatmax (% VO2max, fasted) ----
fatmax_pct_vo2max:
  untrained:     [48, 10]
  recreational:  { M: [51, 8],   F: [50, 10] }
  well_trained:  [56, 8]
  elite:         [60, 7]
  bounds: [25, 80]

# ---- Peak fat oxidation rate (g/min, fasted, normal diet) ----
mfo:
  distribution: lognormal  # very wide IIR
  untrained:     { mu_log: -1.20, sigma_log: 0.40 }  # ~0.30 mean
  recreational:  { M: { mu_log: -0.78, sigma_log: 0.30 },   # ~0.46
                   F: { mu_log: -1.05, sigma_log: 0.33 } }  # ~0.35
  well_trained:  { mu_log: -0.63, sigma_log: 0.30 }  # ~0.53
  elite:         { mu_log: -0.43, sigma_log: 0.27 }  # ~0.65, allow tail to 1.0
  bounds: [0.05, 2.0]
  diet_modifier_keto: 2.3   # multiplicative; Volek 2016

# ---- Crossover (% VO2max), low confidence ----
crossover_pct_vo2max:
  untrained:     [45, 8]
  recreational:  [55, 8]
  well_trained:  [65, 7]
  elite:         [70, 7]

# ---- Cycling gross efficiency (% at 60% VO2max) ----
gross_efficiency:
  untrained:     [19.0, 1.5]
  recreational:  [20.5, 1.5]
  well_trained:  [22.0, 1.3]
  elite:         [23.5, 1.5]
  intensity_slope_pct_per_pct_vo2max: 0.015   # GE rises ~+0.5%-pt from 50->80% VO2max
  bounds: [16, 27]

# ---- Resting RER ----
rer_rest:
  fasted_mixed_diet:   [0.82, 0.04]
  postprandial_2_3h:   [0.88, 0.04]
  keto_adapted:        [0.74, 0.03]
  high_carb:           [0.86, 0.04]
  bounds: [0.70, 0.95]

# ---- RER curve shape (sigmoid on %VO2max) ----
rer_curve:
  form: "RER = R0 + (Rmax - R0) / (1 + exp(-k*(pct_vo2max - x0)))"
  R0:     { rest: 0.82 }     # fasted; override per subject
  Rmax:   [1.10, 0.04]       # exhaustion ceiling
  x0:     "subject.lt2_pct_vo2max"
  k:      { untrained: [0.10, 0.02], trained: [0.13, 0.02] }
  noise_sd_30s_window: 0.015

# ---- Prolonged-exercise RER drift @ ~60% VO2max (fasted, no fuel) ----
prolonged_rer_drift:
  delta_rer_per_min: -0.0006   # ~ -0.07 over 120 min
  glycogen_tau_min:
    untrained:     90
    recreational:  110
    well_trained:  130
    elite:         150
  with_carb_60g_per_hr:
    delta_rer_offset_steady: +0.04
    drift_attenuation: 0.30    # drift slope multiplied by this

# ---- Pre-exercise meal effect ----
preexercise_carb_meal:
  rer_offset_at_60pct_vo2max: +0.04    # vs fasted; sd 0.02

# ---- VO2 kinetics tau (s), phase II ----
vo2_tau_moderate:
  untrained:     [35, 8]
  recreational:  [25, 6]
  well_trained:  [18, 4]
  elite:         [10, 3]
  bounds: [5, 60]
cardiodynamic_delay_s: [18, 4]

# ---- VO2 slow component (mL/min, asymptotic amplitude) ----
vo2_slow_component:
  moderate_below_lt1:    [20, 30]
  heavy_lt1_to_lt2:      [130, 80]
  severe_above_lt2:      [450, 170]
  extreme_near_vo2max:   [1200, 300]
  onset_delay_s:         [100, 30]
  training_reduction:    0.65   # multiply amplitude in well_trained / elite

# ---- HR drift at constant moderate power ----
hr_drift_bpm_per_min_at_65pct_vo2max:
  untrained:     [0.10, 0.04]   # ~6 bpm/h
  recreational:  [0.08, 0.03]
  well_trained:  [0.06, 0.03]
  elite:         [0.05, 0.02]
  heat_multiplier_at_30C_plus: 3.0
  dehydration_multiplier_per_pct_bw_loss: 1.3

# ---- Frayn (1983) / Jeukendrup-Wallis (2005) stoichiometry ----
substrate_equations:
  rest:
    fat_g_per_min: "1.695*VO2 - 1.701*VCO2"      # L/min STPD; N=0
    cho_g_per_min: "4.55*VCO2 - 3.21*VO2"
  exercise:
    fat_g_per_min: "1.695*VO2 - 1.701*VCO2"
    cho_g_per_min: "4.344*VCO2 - 3.061*VO2"
  valid_rer: [0.71, 1.00]
  clamp_fat_at_high_rer: true
  invalidate_above_pct_vo2max: 85   # bicarbonate buffering distorts VCO2

# ---- Sex modifier (apply on top of strata) ----
sex_modifier_F:
  rer_offset_at_matched_pct_vo2max: -0.025   # women slightly lower RER (more fat)
  fatmax_offset_pct_vo2max: +3
```

---

## Key uncertainties / caveats

1. **Crossover concept**: low confidence (C); largely a re-statement of the RER curve. Don't claim it as a measured threshold — derive from the RER curve instead.
2. **MFO**: extremely wide individual range (CV ~30%). Lognormal prior with long right tail is more defensible than Gaussian.
3. **HRmax**: 220−age is biased; use Tanaka. Even Tanaka has ±10 bpm residual SD — apply explicit subject offset.
4. **Above LT2 / 85% VO2max**: do NOT compute fat oxidation from RER. Bicarbonate buffering breaks the assumptions.
5. **Sex differences in absolute MFO are body-comp confounded**: prefer matched fat-free-mass normalization when comparing.
6. **Treadmill vs cycle**: fat ox is ~25% higher on treadmill at matched %VO2max (Achten 2003) — note if you add running.

---

## Key sources

- Tanaka H, Monahan KD, Seals DR (2001). Age-predicted maximal heart rate revisited. *J Am Coll Cardiol* 37:153-156.
- Frayn KN (1983). Calculation of substrate oxidation rates in vivo from gaseous exchange. *J Appl Physiol* 55:628-634.
- Jeukendrup AE, Wallis GA (2005). Measurement of substrate oxidation during exercise by means of gas exchange measurements. *Int J Sports Med* 26 Suppl 1:S28-37.
- Achten J, Jeukendrup AE (2003). Maximal fat oxidation during exercise in trained men. *Int J Sports Med* 24:603-608.
- Maunder E, Plews DJ, Kilding AE (2018). Contextualising maximal fat oxidation during exercise: determinants and normative values. *Front Physiol* 9:599.
- Romijn JA et al. (1993). Regulation of endogenous fat and carbohydrate metabolism in relation to exercise intensity and duration. *Am J Physiol* 265:E380-91.
- Brooks GA, Mercier J (1994). Balance of carbohydrate and lipid utilization during exercise: the "crossover" concept. *J Appl Physiol* 76:2253-61.
- Volek JS et al. (2016). Metabolic characteristics of keto-adapted ultra-endurance runners. *Metabolism* 65:100-110.
- Coyle EF, González-Alonso J (2001). Cardiovascular drift during prolonged exercise. *Exerc Sport Sci Rev* 29:88-92.
- Koppo K, Bouckaert J, Jones AM (2004). Effects of training status and exercise intensity on phase II VO2 kinetics. *Med Sci Sports Exerc* 36:225-32.
- Hopker JG et al. (2010). The effects of training on gross efficiency in cycling. *Int J Sports Med* 31:382-91.
- Lucia A, Hoyos J, Chicharro JL (2001). Physiology of professional road cycling. *Sports Med* 31:325-37.
- Bongers BC et al. / Loe H et al. — LowLands Fitness Registry. *BJSM / Front Physiol*.
- Coyle EF et al. (1986). Muscle glycogen utilization during prolonged strenuous exercise when fed carbohydrate. *J Appl Physiol* 61:165-72.
- ACSM's Guidelines for Exercise Testing and Prescription, 11th ed.
