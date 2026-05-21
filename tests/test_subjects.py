"""Distribution-level tests for the subject sampler."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from bioml.config import load_spec
from bioml.generator.subjects import sample_subjects


@pytest.fixture(scope="module")
def spec() -> dict:
    return load_spec()


@pytest.fixture(scope="module")
def big_cohort(spec: dict) -> list:
    return sample_subjects(spec, n=2000, seed=7)


def test_all_subjects_pass_schema_validation(spec: dict) -> None:
    subjects = sample_subjects(spec, n=200, seed=11)
    assert len(subjects) == 200
    for s in subjects:
        assert 18 <= s.age_years <= 65
        assert s.sex in ("M", "F")
        assert s.training_status in ("untrained", "recreational", "well_trained", "elite")


def test_lt2_always_above_lt1_with_min_gap(big_cohort: list) -> None:
    gaps = np.array(
        [s.lt2_pct_vo2max_true - s.lt1_pct_vo2max_true for s in big_cohort]
    )
    assert (gaps >= 8.0).all()


def test_training_status_proportions_match_weights(
    spec: dict, big_cohort: list
) -> None:
    target = spec["subject"]["training_status"]["weights"]
    counts = Counter(s.training_status for s in big_cohort)
    for stratum, target_p in target.items():
        observed_p = counts[stratum] / len(big_cohort)
        assert abs(observed_p - target_p) < 0.04, f"{stratum}: {observed_p} vs {target_p}"


def test_vo2max_means_by_stratum_match_priors(spec: dict) -> None:
    # Filter to age <= 28 so the age decrement (0.45/yr after age 25) doesn't
    # confound the comparison against the prior's age-25 anchor.
    cohort = sample_subjects(spec, n=8000, seed=17)
    cohort = [s for s in cohort if s.age_years <= 28]
    vo2_cfg = spec["subject"]["vo2max_mlkgmin"]["by_stratum"]
    for stratum in ("untrained", "recreational", "well_trained", "elite"):
        for sex in ("M", "F"):
            bucket = [
                s.vo2max_mlkgmin_true
                for s in cohort
                if s.training_status == stratum and s.sex == sex
            ]
            if len(bucket) < 30:
                continue
            target_mean = vo2_cfg[stratum][sex][0]
            observed_mean = float(np.mean(bucket))
            assert abs(observed_mean - target_mean) < 4.0, (
                f"{stratum}/{sex}: observed {observed_mean:.1f} vs target {target_mean}"
            )


def test_lt1_lt2_correlation_within_subject(big_cohort: list, spec: dict) -> None:
    lt1 = np.array([s.lt1_pct_vo2max_true for s in big_cohort])
    lt2 = np.array([s.lt2_pct_vo2max_true for s in big_cohort])
    target_rho = spec["subject"]["thresholds_pct_vo2max"]["lt1_lt2_correlation"]
    by_stratum_correlations = []
    for stratum in ("untrained", "recreational", "well_trained", "elite"):
        mask = np.array([s.training_status == stratum for s in big_cohort])
        if mask.sum() < 50:
            continue
        r = float(np.corrcoef(lt1[mask], lt2[mask])[0, 1])
        by_stratum_correlations.append(r)
    pooled = float(np.mean(by_stratum_correlations))
    assert abs(pooled - target_rho) < 0.15, f"pooled within-stratum r={pooled:.2f}"


def test_age_decrement_lowers_vo2max(spec: dict) -> None:
    young = sample_subjects(spec, n=500, seed=42)
    old = sample_subjects(spec, n=500, seed=42)
    young_m_rec = [
        s.vo2max_mlkgmin_true
        for s in young
        if s.age_years <= 30 and s.sex == "M" and s.training_status == "recreational"
    ]
    old_m_rec = [
        s.vo2max_mlkgmin_true
        for s in old
        if s.age_years >= 55 and s.sex == "M" and s.training_status == "recreational"
    ]
    if len(young_m_rec) >= 20 and len(old_m_rec) >= 20:
        assert float(np.mean(young_m_rec)) > float(np.mean(old_m_rec))


def test_mfo_lognormal_has_right_skew(big_cohort: list) -> None:
    well_trained = np.array(
        [s.mfo_g_per_min_true for s in big_cohort if s.training_status == "well_trained"]
    )
    assert len(well_trained) > 50
    assert float(np.median(well_trained)) < float(np.mean(well_trained))


def test_keto_subjects_have_higher_mfo(big_cohort: list) -> None:
    keto = np.array([s.mfo_g_per_min_true for s in big_cohort if s.habitual_diet == "keto"])
    mixed = np.array(
        [s.mfo_g_per_min_true for s in big_cohort if s.habitual_diet == "mixed"]
    )
    if len(keto) >= 30 and len(mixed) >= 100:
        assert float(np.mean(keto)) > float(np.mean(mixed))


def test_reproducible_with_seed(spec: dict) -> None:
    a = sample_subjects(spec, n=10, seed=123)
    b = sample_subjects(spec, n=10, seed=123)
    for sa, sb in zip(a, b, strict=True):
        assert sa.vo2max_mlkgmin_true == sb.vo2max_mlkgmin_true
        assert sa.lt1_pct_vo2max_true == sb.lt1_pct_vo2max_true
        assert sa.training_status == sb.training_status
