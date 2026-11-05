"""Evaluation metrics for the multitask model.

Implements, without an sklearn dependency:
- MAE, RMSE, bias (signed mean error), per-intensity-bin MAE and bias
- ROC AUC (Mann-Whitney rank statistic), Brier score, reliability table, ECE
- Substrate-regime metrics: fat/CHO oxidation error inside the Frayn-valid
  envelope plus precision/recall of the inference-time gate itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

INTENSITY_BINS: tuple[float, ...] = (0, 30, 50, 65, 80, 90, 110)

FloatArray = NDArray[np.float64]


def _f64(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.mean(np.abs(_f64(y_true) - _f64(y_pred))))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(np.mean((_f64(y_true) - _f64(y_pred)) ** 2)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error: positive means predictions overshoot truth."""
    return float(np.mean(_f64(y_pred) - _f64(y_true)))


def _by_intensity_bin(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pct_vo2max: ArrayLike,
    bins: tuple[float, ...],
    signed: bool,
) -> list[tuple[str, int, float]]:
    t, p, pct = _f64(y_true), _f64(y_pred), _f64(pct_vo2max)
    out: list[tuple[str, int, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (pct >= lo) & (pct < hi) & np.isfinite(t) & np.isfinite(p)
        n = int(mask.sum())
        label = f"{lo:.0f}-{hi:.0f}%"
        if n == 0:
            out.append((label, 0, float("nan")))
            continue
        err = (p[mask] - t[mask]) if signed else np.abs(t[mask] - p[mask])
        out.append((label, n, float(np.mean(err))))
    return out


def mae_by_intensity_bin(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pct_vo2max: ArrayLike,
    bins: tuple[float, ...] = INTENSITY_BINS,
) -> list[tuple[str, int, float]]:
    """Returns list of (bin_label, n, MAE)."""
    return _by_intensity_bin(y_true, y_pred, pct_vo2max, bins, signed=False)


def bias_by_intensity_bin(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    pct_vo2max: ArrayLike,
    bins: tuple[float, ...] = INTENSITY_BINS,
) -> list[tuple[str, int, float]]:
    """Returns list of (bin_label, n, signed bias)."""
    return _by_intensity_bin(y_true, y_pred, pct_vo2max, bins, signed=True)


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Mann-Whitney U statistic / (n_pos * n_neg). Handles ties via rank averaging."""
    y = np.asarray(y_true, dtype=bool)
    s = _f64(y_score)
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
    return float(np.mean((_f64(y_prob) - _f64(y_true)) ** 2))


def reliability_table(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10
) -> list[tuple[float, float, int, float, float]]:
    """Equal-width probability bins -> (lo, hi, n, mean_predicted, observed_rate)."""
    y, p = _f64(y_true), _f64(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out: list[tuple[float, float, int, float, float]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            out.append((float(edges[b]), float(edges[b + 1]), 0, float("nan"), float("nan")))
        else:
            out.append(
                (
                    float(edges[b]),
                    float(edges[b + 1]),
                    n,
                    float(p[mask].mean()),
                    float(y[mask].mean()),
                )
            )
    return out


def expected_calibration_error(y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10) -> float:
    """Sample-weighted mean |observed rate - mean predicted| across probability bins."""
    table = reliability_table(y_true, y_prob, n_bins)
    total = sum(n for _, _, n, _, _ in table)
    if total == 0:
        return float("nan")
    return float(sum(n / total * abs(obs - pred) for _, _, n, pred, obs in table if n > 0))


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
    rer_bias_by_intensity: list[tuple[str, int, float]] = field(default_factory=list)
    vo2_mae_by_intensity: list[tuple[str, int, float]] = field(default_factory=list)
    vo2_bias_by_intensity: list[tuple[str, int, float]] = field(default_factory=list)
    lt2_ece: float = float("nan")
    lt2_reliability: list[tuple[float, float, int, float, float]] = field(default_factory=list)
    n_rows: int = 0

    def summary(self) -> str:
        lines = [
            f"RER:  MAE={self.rer_mae:.4f}  RMSE={self.rer_rmse:.4f}  bias={self.rer_bias:+.4f}",
            (
                f"VO2:  MAE={self.vo2_mae_mlmin:.1f} mL/min  "
                f"RMSE={self.vo2_rmse_mlmin:.1f}  bias={self.vo2_bias_mlmin:+.1f}"
            ),
            f"LT2:  AUC={self.lt2_auc:.3f}  Brier={self.lt2_brier:.4f}  ECE={self.lt2_ece:.4f}",
            "RER MAE by intensity bin:",
        ]
        for label, n, m in self.rer_mae_by_intensity:
            if n > 0:
                lines.append(f"  {label:>10s}  n={n:>6d}  MAE={m:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "rer_mae": self.rer_mae,
            "rer_rmse": self.rer_rmse,
            "rer_bias": self.rer_bias,
            "vo2_mae_mlmin": self.vo2_mae_mlmin,
            "vo2_rmse_mlmin": self.vo2_rmse_mlmin,
            "vo2_bias_mlmin": self.vo2_bias_mlmin,
            "lt2_auc": self.lt2_auc,
            "lt2_brier": self.lt2_brier,
            "lt2_ece": self.lt2_ece,
            "rer_mae_by_intensity": self.rer_mae_by_intensity,
            "rer_bias_by_intensity": self.rer_bias_by_intensity,
            "vo2_mae_by_intensity": self.vo2_mae_by_intensity,
            "vo2_bias_by_intensity": self.vo2_bias_by_intensity,
            "lt2_reliability": self.lt2_reliability,
        }


def evaluate_multitask(
    rer_true: ArrayLike,
    rer_pred: ArrayLike,
    vo2_true: ArrayLike,
    vo2_pred: ArrayLike,
    lt2_true: ArrayLike,
    lt2_prob: ArrayLike,
    pct_vo2max_true: ArrayLike,
) -> MultitaskReport:
    return MultitaskReport(
        n_rows=int(len(_f64(rer_true))),
        rer_mae=mae(rer_true, rer_pred),
        rer_rmse=rmse(rer_true, rer_pred),
        rer_bias=bias(rer_true, rer_pred),
        rer_mae_by_intensity=mae_by_intensity_bin(rer_true, rer_pred, pct_vo2max_true),
        rer_bias_by_intensity=bias_by_intensity_bin(rer_true, rer_pred, pct_vo2max_true),
        vo2_mae_mlmin=mae(vo2_true, vo2_pred),
        vo2_rmse_mlmin=rmse(vo2_true, vo2_pred),
        vo2_bias_mlmin=bias(vo2_true, vo2_pred),
        vo2_mae_by_intensity=mae_by_intensity_bin(vo2_true, vo2_pred, pct_vo2max_true),
        vo2_bias_by_intensity=bias_by_intensity_bin(vo2_true, vo2_pred, pct_vo2max_true),
        lt2_auc=roc_auc(lt2_true, lt2_prob),
        lt2_brier=brier_score(lt2_true, lt2_prob),
        lt2_ece=expected_calibration_error(lt2_true, lt2_prob),
        lt2_reliability=reliability_table(lt2_true, lt2_prob),
    )


@dataclass(frozen=True)
class SubstrateReport:
    """Error of the downstream fat/CHO emission, plus the quality of the gate.

    Truth validity = the generator (or lab) produced a Frayn-valid fat_ox value.
    Predicted validity = the inference gate chose ``substrate_calculable``.
    """

    n_rows: int
    n_truth_valid: int
    n_pred_valid: int
    n_both_valid: int
    fat_ox_mae_g_per_min: float
    fat_ox_bias_g_per_min: float
    cho_ox_mae_g_per_min: float
    cho_ox_bias_g_per_min: float
    gate_precision: float
    gate_recall: float
    gate_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def evaluate_substrate(
    fat_true: ArrayLike,
    cho_true: ArrayLike,
    fat_pred: ArrayLike,
    cho_pred: ArrayLike,
    pred_valid: ArrayLike,
) -> SubstrateReport:
    ft, ct, fp, cp = _f64(fat_true), _f64(cho_true), _f64(fat_pred), _f64(cho_pred)
    pv = np.asarray(pred_valid, dtype=bool)
    tv = np.isfinite(ft)
    both = tv & pv & np.isfinite(fp)
    n_both = int(both.sum())
    tp = n_both
    precision = tp / pv.sum() if pv.sum() else float("nan")
    recall = tp / tv.sum() if tv.sum() else float("nan")
    accuracy = float(np.mean(pv == tv))
    nan = float("nan")
    return SubstrateReport(
        n_rows=int(len(ft)),
        n_truth_valid=int(tv.sum()),
        n_pred_valid=int(pv.sum()),
        n_both_valid=n_both,
        fat_ox_mae_g_per_min=mae(ft[both], fp[both]) if n_both else nan,
        fat_ox_bias_g_per_min=bias(ft[both], fp[both]) if n_both else nan,
        cho_ox_mae_g_per_min=mae(ct[both], cp[both]) if n_both else nan,
        cho_ox_bias_g_per_min=bias(ct[both], cp[both]) if n_both else nan,
        gate_precision=float(precision),
        gate_recall=float(recall),
        gate_accuracy=accuracy,
    )
