"""End-to-end test: generate a small dataset and verify round-trip."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from bioml.generator.run import generate_dataset


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bioml_dataset")
    generate_dataset(n_subjects=5, out_dir=out, seed=123, log_every=0)
    return out


def test_top_level_parquets_exist(small_dataset: Path) -> None:
    assert (small_dataset / "subjects.parquet").exists()
    assert (small_dataset / "sessions.parquet").exists()
    assert (small_dataset / "samples").is_dir()


def test_subjects_parquet_row_count(small_dataset: Path) -> None:
    df = pl.read_parquet(small_dataset / "subjects.parquet")
    assert df.height == 5
    expected_cols = {
        "subject_id",
        "age_years",
        "sex",
        "training_status",
        "vo2max_mlkgmin_true",
        "lt1_pct_vo2max_true",
        "lt2_pct_vo2max_true",
        "mfo_g_per_min_true",
    }
    assert expected_cols <= set(df.columns)


def test_sessions_parquet_row_count(small_dataset: Path) -> None:
    df = pl.read_parquet(small_dataset / "sessions.parquet")
    assert df.height == 5


def test_samples_partitioning_by_subject(small_dataset: Path) -> None:
    partitions = sorted((small_dataset / "samples").iterdir())
    assert len(partitions) == 5
    for p in partitions:
        assert p.is_dir()
        assert p.name.startswith("subject_id=")
        files = list(p.glob("*.parquet"))
        assert len(files) == 1


def test_per_session_samples_roundtrip(small_dataset: Path) -> None:
    sessions_df = pl.read_parquet(small_dataset / "sessions.parquet")
    for row in sessions_df.iter_rows(named=True):
        subject_id = row["subject_id"]
        session_id = row["session_id"]
        duration_s = row["duration_s"]
        session_file = (
            small_dataset / "samples" / f"subject_id={subject_id}" / f"{session_id}.parquet"
        )
        assert session_file.exists()
        samples = pl.read_parquet(session_file)
        assert samples.height == duration_s
        assert "rer_lab" in samples.columns
        assert "hr_wearable_bpm" in samples.columns
        assert "above_lt2_flag" in samples.columns


def test_above_lt2_flag_aligns_with_rer(small_dataset: Path) -> None:
    sessions_df = pl.read_parquet(small_dataset / "sessions.parquet")
    first = sessions_df.row(0, named=True)
    samples = pl.read_parquet(
        small_dataset / "samples" / f"subject_id={first['subject_id']}" / f"{first['session_id']}.parquet"
    )
    high_rer = samples.filter(pl.col("rer_lab") >= 1.0)
    if high_rer.height > 0:
        assert high_rer["above_lt2_flag"].all()


def test_frayn_columns_nan_above_lt2(small_dataset: Path) -> None:
    sessions_df = pl.read_parquet(small_dataset / "sessions.parquet")
    first = sessions_df.row(0, named=True)
    samples = pl.read_parquet(
        small_dataset / "samples" / f"subject_id={first['subject_id']}" / f"{first['session_id']}.parquet"
    )
    above = samples.filter(pl.col("above_lt2_flag"))
    if above.height > 0:
        assert above["fat_ox_g_per_min"].is_null().all()
        assert above["cho_ox_g_per_min"].is_null().all()


def test_hive_partitioned_scan_works(small_dataset: Path) -> None:
    """Verify the hive-partitioned scan recovers all sessions via a single query."""
    all_samples = pl.scan_parquet(
        small_dataset / "samples" / "**" / "*.parquet",
        hive_partitioning=True,
    ).collect()
    expected_rows = pl.read_parquet(small_dataset / "sessions.parquet")["duration_s"].sum()
    assert all_samples.height == expected_rows
    assert "subject_id" in all_samples.columns
    assert all_samples["subject_id"].n_unique() == 5
