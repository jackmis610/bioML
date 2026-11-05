"""LightGBM multitask trainer: predict RER, VO2, and above-LT2 from wearable inputs.

Three boosters share one wearable-only feature matrix. Huber loss on RER
(δ=0.02, design decision D4), squared error on VO2 (design decision D5; see
``DEFAULT_VO2_PARAMS``), binary cross-entropy with positive-class weight 4 on
the LT2 flag.

The saved artifact is a directory with three LightGBM text models plus a
``metadata.json`` that pins the feature contract, hyperparameters, data
provenance, and the cross-validated metrics the model shipped with. ``predict``
refuses a feature frame whose columns do not match the contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import polars as pl
from numpy.typing import NDArray

from bioml import __version__
from bioml.eval.features import FEATURE_COLS, TrainingData, build_training_data
from bioml.eval.loaders import Dataset, load_dataset
from bioml.eval.metrics import MultitaskReport, evaluate_multitask
from bioml.eval.splits import subject_kfold

METADATA_FILE = "metadata.json"
HEAD_FILES = {"rer": "rer.txt", "vo2": "vo2.txt", "lt2": "lt2.txt"}

DEFAULT_RER_PARAMS: dict[str, Any] = {
    "objective": "huber",
    "alpha": 0.02,
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Squared error, not Huber (design decision D5). Huber with delta = 50 mL/min clips the
# gradient at 50 for every residual larger than that, so 400 rounds at lr 0.05 could
# never catch up at the top of the intensity range: VO2 bias was -592 mL/min in the
# 90-110% VO2max bin. L2 removed the bias (-5 mL/min overall) and cut MAE 181 -> 113.
DEFAULT_VO2_PARAMS: dict[str, Any] = {
    k: v for k, v in DEFAULT_RER_PARAMS.items() if k != "alpha"
} | {"objective": "regression", "metric": "l2"}

DEFAULT_LT2_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": 4.0,
    "verbose": -1,
}

FloatArray = NDArray[np.float64]


class Predictor(Protocol):
    """Minimal fit/predict interface shared by the ML model and the baselines."""

    def fit(self, features: pl.DataFrame, targets: pl.DataFrame) -> Predictor: ...

    def predict(self, features: pl.DataFrame) -> dict[str, FloatArray]: ...


class FeatureContractError(ValueError):
    """Raised when a feature frame does not match the columns the model was trained on."""


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


@dataclass(frozen=True)
class TrainedModel:
    rer: lgb.Booster
    vo2: lgb.Booster
    lt2: lgb.Booster
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _matrix(self, features: pl.DataFrame) -> FloatArray:
        cols = tuple(features.columns)
        if cols != self.feature_names:
            missing = [c for c in self.feature_names if c not in cols]
            extra = [c for c in cols if c not in self.feature_names]
            raise FeatureContractError(
                "feature frame does not match the model's contract: "
                f"missing={missing} extra={extra} "
                f"(order must also match; expected {list(self.feature_names)})"
            )
        return features.cast(pl.Float64).to_numpy()

    def predict(self, features: pl.DataFrame) -> dict[str, FloatArray]:
        x = self._matrix(features)
        return {
            "rer": np.asarray(self.rer.predict(x), dtype=np.float64),
            "vo2_mlmin": np.asarray(self.vo2.predict(x), dtype=np.float64),
            "lt2_prob": np.asarray(self.lt2.predict(x), dtype=np.float64),
        }

    def shap_contributions(self, features: pl.DataFrame) -> dict[str, FloatArray]:
        """Per-row SHAP values from LightGBM's native TreeSHAP (``pred_contrib``).

        Returns arrays of shape (n_rows, n_features); the bias column is dropped.
        """
        x = self._matrix(features)
        out: dict[str, FloatArray] = {}
        for name, booster in (("rer", self.rer), ("vo2", self.vo2), ("lt2", self.lt2)):
            contrib = np.asarray(booster.predict(x, pred_contrib=True), dtype=np.float64)
            out[name] = contrib[:, :-1]
        return out

    def feature_importance(self, importance_type: str = "gain") -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, booster in (("rer", self.rer), ("vo2", self.vo2), ("lt2", self.lt2)):
            imp = booster.feature_importance(importance_type=importance_type)
            total = float(imp.sum()) or 1.0
            out[name] = {f: float(v) / total for f, v in zip(self.feature_names, imp, strict=True)}
        return out

    def save(self, path: Path | str, extra_metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.rer.save_model(str(path / HEAD_FILES["rer"]))
        self.vo2.save_model(str(path / HEAD_FILES["vo2"]))
        self.lt2.save_model(str(path / HEAD_FILES["lt2"]))
        meta = {
            **self.metadata,
            **(extra_metadata or {}),
            "feature_names": list(self.feature_names),
            "n_features": len(self.feature_names),
            "heads": list(HEAD_FILES),
            "bioml_version": __version__,
            "lightgbm_version": lgb.__version__,
            "saved_at": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
        }
        (path / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path | str) -> TrainedModel:
        path = Path(path)
        meta_path = path / METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing: model directories must carry metadata.json "
                "(retrain with `python -m bioml train --save-model`)."
            )
        meta = json.loads(meta_path.read_text())
        return cls(
            rer=lgb.Booster(model_file=str(path / HEAD_FILES["rer"])),
            vo2=lgb.Booster(model_file=str(path / HEAD_FILES["vo2"])),
            lt2=lgb.Booster(model_file=str(path / HEAD_FILES["lt2"])),
            feature_names=tuple(meta["feature_names"]),
            metadata=meta,
        )


def fit_multitask(
    train_data: TrainingData,
    n_estimators: int = 300,
    rer_params: dict[str, Any] | None = None,
    vo2_params: dict[str, Any] | None = None,
    lt2_params: dict[str, Any] | None = None,
) -> TrainedModel:
    feature_names = tuple(train_data.features.columns)
    x = train_data.features.cast(pl.Float64).to_numpy()
    y_rer = train_data.targets["rer_lab"].to_numpy()
    y_vo2 = train_data.targets["vo2_lab_mlmin"].to_numpy()
    y_lt2 = train_data.targets["above_lt2_flag"].to_numpy().astype(np.int64)
    names = list(feature_names)

    rer_p = rer_params or DEFAULT_RER_PARAMS
    vo2_p = vo2_params or DEFAULT_VO2_PARAMS
    lt2_p = lt2_params or DEFAULT_LT2_PARAMS

    rer_model = lgb.train(rer_p, lgb.Dataset(x, label=y_rer, feature_name=names), n_estimators)
    vo2_model = lgb.train(vo2_p, lgb.Dataset(x, label=y_vo2, feature_name=names), n_estimators)
    lt2_model = lgb.train(lt2_p, lgb.Dataset(x, label=y_lt2, feature_name=names), n_estimators)

    metadata = {
        "model_family": "lightgbm_multitask",
        "n_estimators": n_estimators,
        "params": {"rer": rer_p, "vo2": vo2_p, "lt2": lt2_p},
        "n_train_rows": int(x.shape[0]),
        "n_train_subjects": int(train_data.groups.n_unique()),
        "lt2_positive_rate_train": float(y_lt2.mean()),
    }
    return TrainedModel(
        rer=rer_model, vo2=vo2_model, lt2=lt2_model, feature_names=feature_names, metadata=metadata
    )


@dataclass
class LightGBMMultitask:
    """Predictor-protocol wrapper so the ML model and baselines run through one CV loop."""

    n_estimators: int = 300
    model: TrainedModel | None = None

    def fit(self, features: pl.DataFrame, targets: pl.DataFrame) -> LightGBMMultitask:
        td = TrainingData(
            features=features,
            targets=targets,
            groups=pl.Series("subject_id", [""] * features.height),
            keys=pl.DataFrame(),
        )
        self.model = fit_multitask(td, n_estimators=self.n_estimators)
        return self

    def predict(self, features: pl.DataFrame) -> dict[str, FloatArray]:
        if self.model is None:
            raise RuntimeError("fit before predict")
        return self.model.predict(features)


@dataclass
class CVRun:
    """Out-of-fold results for one predictor over subject-wise folds."""

    name: str
    fold_reports: list[MultitaskReport]
    oof: pl.DataFrame  # keys + targets + eval cols + predictions, all folds concatenated
    shap_mean_abs: dict[str, dict[str, float]] | None = None
    fit_seconds: float = 0.0

    def mean(self, attr: str) -> float:
        return float(np.nanmean([getattr(r, attr) for r in self.fold_reports]))

    def sd(self, attr: str) -> float:
        return float(np.nanstd([getattr(r, attr) for r in self.fold_reports]))


def run_cv(
    ds: Dataset,
    make_predictor: Callable[[], Predictor],
    name: str,
    k: int = 5,
    seed: int = 42,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
    compute_shap: bool = False,
    verbose: bool = False,
) -> CVRun:
    """Subject-wise K-fold for any Predictor; returns per-fold reports and OOF predictions."""
    import time

    subject_ids = ds.subjects["subject_id"].to_list()
    splits = subject_kfold(subject_ids, k=k, seed=seed)

    fold_reports: list[MultitaskReport] = []
    oof_parts: list[pl.DataFrame] = []
    shap_acc: dict[str, FloatArray] = {}
    shap_n = 0
    fit_seconds = 0.0

    for fold_i, (train_ids, val_ids) in enumerate(splits):
        train_td = build_training_data(
            ds.samples, ds.subjects, ds.sessions, subject_ids=train_ids, feature_cols=feature_cols
        )
        val_td = build_training_data(
            ds.samples, ds.subjects, ds.sessions, subject_ids=val_ids, feature_cols=feature_cols
        )
        t0 = time.perf_counter()
        predictor = make_predictor().fit(train_td.features, train_td.targets)
        fit_seconds += time.perf_counter() - t0
        preds = predictor.predict(val_td.features)

        if val_td.eval_cols is None or "pct_vo2max_lab" not in val_td.eval_cols.columns:
            raise ValueError("evaluation requires pct_vo2max_lab in the dataset samples")

        report = evaluate_multitask(
            val_td.targets["rer_lab"].to_numpy(),
            preds["rer"],
            val_td.targets["vo2_lab_mlmin"].to_numpy(),
            preds["vo2_mlmin"],
            val_td.targets["above_lt2_flag"].to_numpy(),
            preds["lt2_prob"],
            val_td.eval_cols["pct_vo2max_lab"].to_numpy(),
        )
        fold_reports.append(report)

        oof_parts.append(
            pl.concat(
                [
                    val_td.keys,
                    val_td.targets,
                    val_td.eval_cols,
                    pl.DataFrame(
                        {
                            "fold": np.full(val_td.n_rows, fold_i, dtype=np.int64),
                            "rer_pred": preds["rer"],
                            "vo2_pred_mlmin": preds["vo2_mlmin"],
                            "lt2_prob": preds["lt2_prob"],
                        }
                    ),
                ],
                how="horizontal",
            )
        )

        if compute_shap and isinstance(predictor, LightGBMMultitask) and predictor.model:
            contribs = predictor.model.shap_contributions(val_td.features)
            for head, arr in contribs.items():
                summed = np.abs(arr).sum(axis=0)
                shap_acc[head] = shap_acc.get(head, np.zeros_like(summed)) + summed
            shap_n += val_td.n_rows

        if verbose:
            print(
                f"  [{name}] fold {fold_i + 1}: RER MAE={report.rer_mae:.4f}  "
                f"VO2 MAE={report.vo2_mae_mlmin:.0f}  LT2 AUC={report.lt2_auc:.3f}"
            )

    shap_mean_abs = None
    if compute_shap and shap_n:
        shap_mean_abs = {
            head: {f: float(v / shap_n) for f, v in zip(feature_cols, arr, strict=True)}
            for head, arr in shap_acc.items()
        }

    return CVRun(
        name=name,
        fold_reports=fold_reports,
        oof=pl.concat(oof_parts, how="vertical"),
        shap_mean_abs=shap_mean_abs,
        fit_seconds=fit_seconds,
    )


def _aggregate_bins(reports: list[MultitaskReport], attr: str) -> list[tuple[str, int, float]]:
    rows = [getattr(r, attr) for r in reports]
    n_bins = len(rows[0])
    out = []
    for bin_i in range(n_bins):
        label = rows[0][bin_i][0]
        vals = [r[bin_i][2] for r in rows]
        ns = [r[bin_i][1] for r in rows]
        out.append((label, int(np.mean(ns)), float(np.nanmean(vals))))
    return out


def summarize_cv(run: CVRun) -> dict[str, Any]:
    r = run.fold_reports
    return {
        "name": run.name,
        "k": len(r),
        "rer_mae": run.mean("rer_mae"),
        "rer_mae_sd": run.sd("rer_mae"),
        "rer_rmse": run.mean("rer_rmse"),
        "rer_bias": run.mean("rer_bias"),
        "vo2_mae_mlmin": run.mean("vo2_mae_mlmin"),
        "vo2_mae_mlmin_sd": run.sd("vo2_mae_mlmin"),
        "vo2_rmse_mlmin": run.mean("vo2_rmse_mlmin"),
        "vo2_bias_mlmin": run.mean("vo2_bias_mlmin"),
        "lt2_auc": run.mean("lt2_auc"),
        "lt2_auc_sd": run.sd("lt2_auc"),
        "lt2_brier": run.mean("lt2_brier"),
        "lt2_ece": run.mean("lt2_ece"),
        "rer_mae_by_intensity": _aggregate_bins(r, "rer_mae_by_intensity"),
        "rer_bias_by_intensity": _aggregate_bins(r, "rer_bias_by_intensity"),
        "vo2_mae_by_intensity": _aggregate_bins(r, "vo2_mae_by_intensity"),
        "vo2_bias_by_intensity": _aggregate_bins(r, "vo2_bias_by_intensity"),
        "fit_seconds": run.fit_seconds,
    }


def cross_validate(
    dataset_path: Path | str,
    k: int = 5,
    seed: int = 42,
    n_estimators: int = 300,
    verbose: bool = True,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
) -> dict[str, Any]:
    """Convenience wrapper: subject-wise CV of the LightGBM model, summary dict + fold reports."""
    ds = load_dataset(dataset_path)
    run = run_cv(
        ds,
        lambda: LightGBMMultitask(n_estimators=n_estimators),
        name="lightgbm_multitask",
        k=k,
        seed=seed,
        feature_cols=feature_cols,
        verbose=verbose,
    )
    out = summarize_cv(run)
    out["fold_reports"] = run.fold_reports
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bioml.train")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/v1"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="If set, also train on all data and save the model (with metadata) here.",
    )
    args = p.parse_args(argv)

    print(f"LightGBM multitask CV: {args.data}")
    print(
        f"  {args.k}-fold subject-wise, {args.n_estimators} boosting rounds, "
        f"{len(FEATURE_COLS)} features"
    )
    print()
    res = cross_validate(args.data, k=args.k, seed=args.seed, n_estimators=args.n_estimators)
    print()
    print("=== LightGBM multitask result ===")
    print(f"  RER MAE     : {res['rer_mae']:.4f} ± {res['rer_mae_sd']:.4f}")
    print(f"  RER RMSE    : {res['rer_rmse']:.4f}")
    print(f"  RER bias    : {res['rer_bias']:+.4f}")
    print(f"  VO2 MAE     : {res['vo2_mae_mlmin']:.0f} ± {res['vo2_mae_mlmin_sd']:.0f} mL/min")
    print(f"  VO2 bias    : {res['vo2_bias_mlmin']:+.0f} mL/min")
    print(f"  LT2 AUC     : {res['lt2_auc']:.3f}")
    print(f"  LT2 Brier   : {res['lt2_brier']:.4f}")
    print(f"  LT2 ECE     : {res['lt2_ece']:.4f}")
    print("  RER MAE / bias by intensity bin:")
    for (label, n, m), (_, _, b) in zip(
        res["rer_mae_by_intensity"], res["rer_bias_by_intensity"], strict=True
    ):
        if n > 0:
            print(f"    {label:>10s}  n_avg={n:>6d}  MAE={m:.4f}  bias={b:+.4f}")
    print("  VO2 MAE / bias by intensity bin (mL/min):")
    for (label, n, m), (_, _, b) in zip(
        res["vo2_mae_by_intensity"], res["vo2_bias_by_intensity"], strict=True
    ):
        if n > 0:
            print(f"    {label:>10s}  n_avg={n:>6d}  MAE={m:.0f}  bias={b:+.0f}")

    if args.save_model is not None:
        print()
        print(f"Training final model on all data and saving to {args.save_model}")
        ds = load_dataset(args.data)
        td = build_training_data(ds.samples, ds.subjects, ds.sessions)
        final_model = fit_multitask(td, n_estimators=args.n_estimators)
        cv_meta = {k_: v for k_, v in res.items() if k_ != "fold_reports"}
        final_model.save(
            args.save_model,
            extra_metadata={
                "training_data": str(args.data),
                "cv": {"k": args.k, "seed": args.seed, **cv_meta},
            },
        )
        print("Saved.")


if __name__ == "__main__":
    main()
