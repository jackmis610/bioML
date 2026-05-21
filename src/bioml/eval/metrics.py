"""Evaluation metrics for the multitask model.

Implements:
- MAE, RMSE, bias (signed mean error)
- Intensity-binned MAE
- AUC and Brier score for the LT2 binary flag

No sklearn dependency: AUC via Mann-Whitney rank statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error: positive means predictions overshoot truth."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(p - t))


def mae_by_intensity_bin(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pct_vo2max: ArrayLike,
    bins: tuple[float, ...] = (0, 30, 50, 65, 80, 90, 110),
) -> list[tuple[str, int, float]]:
    """Returns list of (bin_label, n, MAE)."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    pct = np.asarray(pct_vo2max, dtype=np.float64)
    out: list[tuple[str, int, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (pct >= lo) & (pct < hi) & np.isfinite(t) & np.isfinite(p)
        n = int(mask.sum())
        if n == 0:
            out.append((f"{lo:.0f}-{hi:.0f}%", 0, float("nan")))
        else:
            out.append((f"{lo:.0f}-{hi:.0f}%", n, float(np.mean(np.abs(t[mask] - p[mask])))))
    return out


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Mann-Whitney U statistic / (n_pos * n_neg). Handles ties via rank averaging."""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(ranks[y].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class MultitaskReport:
    rer_mae: float
    rer_rmse: float
    rer_bias: float
    rer_mae_by_intensity: list[tuple[str, int, float]]
    vo2_mae_mlmin: float
    vo2_rmse_mlmin: float
    vo2_bias_mlmin: float
    lt2_auc: float
    lt2_brier: float

    def summary(self) -> str:
        lines = [
            f"RER:  MAE={self.rer_mae:.4f}  RMSE={self.rer_rmse:.4f}  bias={self.rer_bias:+.4f}",
            f"VO2:  MAE={self.vo2_mae_mlmin:.1f} mL/min  RMSE={self.vo2_rmse_mlmin:.1f}  bias={self.vo2_bias_mlmin:+.1f}",
            f"LT2:  AUC={self.lt2_auc:.3f}  Brier={self.lt2_brier:.4f}",
            "RER MAE by intensity bin:",
        ]
        for label, n, m in self.rer_mae_by_intensity:
            if n > 0:
                lines.append(f"  {label:>10s}  n={n:>6d}  MAE={m:.4f}")
        return "\n".join(lines)


def evaluate_multitask(
    rer_true: NDArray,
    rer_pred: NDArray,
    vo2_true: NDArray,
    vo2_pred: NDArray,
    lt2_true: NDArray,
    lt2_prob: NDArray,
    pct_vo2max_true: NDArray,
) -> MultitaskReport:
    return MultitaskReport(
        rer_mae=mae(rer_true, rer_pred),
        rer_rmse=rmse(rer_true, rer_pred),
        rer_bias=bias(rer_true, rer_pred),
        rer_mae_by_intensity=mae_by_intensity_bin(rer_true, rer_pred, pct_vo2max_true),
        vo2_mae_mlmin=mae(vo2_true, vo2_pred),
        vo2_rmse_mlmin=rmse(vo2_true, vo2_pred),
        vo2_bias_mlmin=bias(vo2_true, vo2_pred),
        lt2_auc=roc_auc(lt2_true, lt2_prob),
        lt2_brier=brier_score(lt2_true, lt2_prob),
    )
