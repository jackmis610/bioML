"""Context imputation from population priors (spec.yaml)."""

from __future__ import annotations

import pytest

from bioml.intake import (
    default_hr_rest_bpm,
    estimate_ffm_kg,
    infer_training_status,
    resolve_subject,
)


@pytest.mark.parametrize(
    ("vo2", "sex", "age", "expected"),
    [
        (38.0, "M", 25, "untrained"),
        (50.0, "M", 25, "recreational"),
        (62.0, "M", 25, "well_trained"),
        (78.0, "M", 25, "elite"),
        (42.0, "F", 25, "recreational"),
        (70.0, "F", 25, "elite"),
    ],
)
def test_infer_training_status_hits_stratum_means(
    vo2: float, sex: str, age: int, expected: str
) -> None:
    assert infer_training_status(vo2, sex, age) == expected  # type: ignore[arg-type]


def test_infer_training_status_is_age_adjusted() -> None:
    # 38.75 at age 50 projects to 50.0 at age 25 (0.45/yr) -> recreational, not untrained
    assert infer_training_status(38.75, "M", 50) == "recreational"
    assert infer_training_status(38.75, "M", 25) == "untrained"


def test_estimate_ffm_matches_generator_assumption() -> None:
    assert estimate_ffm_kg(80.0, "M", "recreational") == pytest.approx(80 * 0.82, abs=0.05)
    assert estimate_ffm_kg(80.0, "M", "elite") == pytest.approx(80 * 0.87, abs=0.05)
    assert estimate_ffm_kg(60.0, "F", "untrained") == pytest.approx(60 * 0.74, abs=0.05)


def test_default_hr_rest_from_spec() -> None:
    assert default_hr_rest_bpm("M", "untrained") == 70.0
    assert default_hr_rest_bpm("F", "untrained") == 74.0
    assert default_hr_rest_bpm("M", "elite") == 42.0


def test_resolve_subject_logs_every_imputation() -> None:
    r = resolve_subject(
        age_years=30,
        sex="M",
        height_cm=180.0,
        weight_kg=75.0,
        vo2max_estimated_wearable_mlkgmin=50.0,
    )
    assert set(r.imputed) == {"training_status", "ffm_kg", "hr_rest_bpm"}
    assert r.training_status == "recreational"  # 50 @ age 30 -> 52.25 @ age 25
    assert r.ffm_kg > 0 and r.hr_rest_bpm > 0

    full = resolve_subject(
        age_years=30,
        sex="M",
        height_cm=180.0,
        weight_kg=75.0,
        vo2max_estimated_wearable_mlkgmin=55.0,
        ffm_kg=62.0,
        hr_rest_bpm=48.0,
        training_status="well_trained",
    )
    assert full.imputed == {}
    assert full.ffm_kg == 62.0
