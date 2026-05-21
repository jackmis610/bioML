"""Load a Hive-partitioned synthetic or real dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Dataset:
    subjects: pl.DataFrame
    sessions: pl.DataFrame
    samples: pl.LazyFrame


def load_dataset(path: Path | str) -> Dataset:
    """Load subjects + sessions (eager) and samples (lazy, hive-partitioned)."""
    p = Path(path)
    subjects = pl.read_parquet(p / "subjects.parquet")
    sessions = pl.read_parquet(p / "sessions.parquet")
    samples = pl.scan_parquet(
        p / "samples" / "**" / "*.parquet",
        hive_partitioning=True,
    )
    return Dataset(subjects=subjects, sessions=sessions, samples=samples)
