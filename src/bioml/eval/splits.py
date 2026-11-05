"""Subject-wise cross-validation splits.

The only legitimate split for this project: each subject is fully in train or
fully in test. Time-based or stratified row-shuffles leak subject identity into
the model.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def subject_kfold(
    subject_ids: list[str],
    k: int = 5,
    seed: int = 42,
    rng: Generator | None = None,
) -> list[tuple[list[str], list[str]]]:
    """Return K (train_subjects, val_subjects) tuples.

    Subjects are shuffled deterministically by `seed` and partitioned into K
    contiguous folds.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if len(subject_ids) < k:
        raise ValueError(f"need at least {k} subjects, got {len(subject_ids)}")

    rng = rng if rng is not None else np.random.default_rng(seed)
    ids = list(subject_ids)
    rng.shuffle(ids)

    folds: list[list[str]] = [[] for _ in range(k)]
    for i, sid in enumerate(ids):
        folds[i % k].append(sid)

    splits: list[tuple[list[str], list[str]]] = []
    for i in range(k):
        val = folds[i]
        train = [s for j in range(k) if j != i for s in folds[j]]
        splits.append((train, val))
    return splits


def physiology_stratified_split(
    subject_ids: list[str],
    training_status: list[str],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Hold-out split that ensures all training-status strata are represented in test."""
    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[str]] = {}
    for sid, stratum in zip(subject_ids, training_status, strict=True):
        by_stratum.setdefault(stratum, []).append(sid)

    train: list[str] = []
    test: list[str] = []
    for ids in by_stratum.values():
        ids_arr = np.array(ids)
        rng.shuffle(ids_arr)
        n_test = max(1, int(round(len(ids_arr) * test_fraction)))
        test.extend(ids_arr[:n_test].tolist())
        train.extend(ids_arr[n_test:].tolist())
    return train, test
