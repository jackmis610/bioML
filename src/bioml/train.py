"""LightGBM multitask trainer: predict RER, VO2, and above_LT2 from wearable inputs.

Three separate boosters share the wearable-only feature set. Huber loss on
RER (δ=0.02) and VO2 (δ=50 mL/min) per the Q4 decision; BCE with positive-class
weight 4 on the LT2 flag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from numpy.typing import NDArray

from bioml.eval.features import TrainingData, build_training_data
from bioml.eval.loaders import load_dataset
from bioml.eval.metrics import MultitaskReport, evaluate_multitask
from bioml.eval.splits import subject_kfold

VO2_REST_ML_PER_KG_MIN = 3.5

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

DEFAULT_VO2_PARAMS: dict[str, Any] = {
    **DEFAULT_RER_PARAMS,
    "alpha": 50.0,
}

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


@dataclass(frozen=True)
class TrainedModel:
    rer: lgb.Booster
    vo2: lgb.Booster
    lt2: lgb.Booster

    def predict(self, features: pl.DataFrame) -> dict[str, NDArray[np.float64]]:
        X = features.to_numpy()
        return {
            "rer": self.rer.predict(X),  # type: ignore[return-value]
            "vo2_mlmin": self.vo2.predict(X),  # type: ignore[return-value]
            "lt2_prob": self.lt2.predict(X),  # type: ignore[return-value]
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.rer.save_model(str(path / "rer.txt"))
        self.vo2.save_model(str(path / "vo2.txt"))
        self.lt2.save_model(str(path / "lt2.txt"))

    @classmethod
    def load(cls, path: Path) -> "TrainedModel":
        path = Path(path)
        return cls(
            rer=lgb.Booster(model_file=str(path / "rer.txt")),
            vo2=lgb.Booster(model_file=str(path / "vo2.txt")),
            lt2=lgb.Booster(model_file=str(path / "lt2.txt")),
        )


def fit_multitask(
    train_data: TrainingData,
    n_estimators: int = 300,
    rer_params: dict[str, Any] | None = None,
    vo2_params: dict[str, Any] | None = None,
    lt2_params: dict[str, Any] | None = None,
) -> TrainedModel:
    X = train_data.features.to_numpy()
    y_rer = train_data.targets["rer_lab"].to_numpy()
    y_vo2 = train_data.targets["vo2_lab_mlmin"].to_numpy()
    y_lt2 = train_data.targets["above_lt2_flag"].to_numpy().astype(np.int64)

    rer_model = lgb.train(
        rer_params or DEFAULT_RER_PARAMS,
        lgb.Dataset(X, label=y_rer),
        num_boost_round=n_estimators,
    )
    vo2_model = lgb.train(
        vo2_params or DEFAULT_VO2_PARAMS,
        lgb.Dataset(X, label=y_vo2),
        num_boost_round=n_estimators,
    )
    lt2_model = lgb.train(
        lt2_params or DEFAULT_LT2_PARAMS,
        lgb.Dataset(X, label=y_lt2),
        num_boost_round=n_estimators,
    )
    return TrainedModel(rer=rer_model, vo2=vo2_model, lt2=lt2_model)


def _pct_vo2max_from_features_and_truth(
    features: pl.DataFrame, vo2_true_mlmin: NDArray[np.float64]
) -> NDArray[np.float64]:
    weight = features["weight_kg"].to_numpy()
    vo2max_est = features["vo2max_estimated_wearable_mlkgmin"].to_numpy()
    vo2_rest = VO2_REST_ML_PER_KG_MIN * weight
    vo2max_abs = vo2max_est * weight
    return (vo2_true_mlmin - vo2_rest) / np.maximum(vo2max_abs - vo2_rest, 1.0) * 100.0


def cross_validate(
    dataset_path: Path,
    k: int = 5,
    seed: int = 42,
    n_estimators: int = 300,
    verbose: bool = True,
) -> dict[str, Any]:
    ds = load_dataset(dataset_path)
    subject_ids = ds.subjects["subject_id"].to_list()
    splits = subject_kfold(subject_ids, k=k, seed=seed)

    fold_reports: list[MultitaskReport] = []
    for fold_i, (train_ids, val_ids) in enumerate(splits):
        train_td = build_training_data(
            ds.samples, ds.subjects, ds.sessions, subject_ids=train_ids
        )
        val_td = build_training_data(
            ds.samples, ds.subjects, ds.sessions, subject_ids=val_ids
        )
        model = fit_multitask(train_td, n_estimators=n_estimators)
        preds = model.predict(val_td.features)

        rer_true = val_td.targets["rer_lab"].to_numpy()
        vo2_true = val_td.targets["vo2_lab_mlmin"].to_numpy()
        lt2_true = val_td.targets["above_lt2_flag"].to_numpy()
        pct_true = _pct_vo2max_from_features_and_truth(val_td.features, vo2_true)

        report = evaluate_multitask(
            rer_true,
            preds["rer"],
            vo2_true,
            preds["vo2_mlmin"],
            lt2_true,
            preds["lt2_prob"],
            pct_true,
        )
        fold_reports.append(report)
        if verbose:
            print(
                f"  fold {fold_i + 1}: RER MAE={report.rer_mae:.4f}  "
                f"VO2 MAE={report.vo2_mae_mlmin:.0f}  LT2 AUC={report.lt2_auc:.3f}"
            )

    return {
        "fold_reports": fold_reports,
        "rer_mae": float(np.mean([r.rer_mae for r in fold_reports])),
        "rer_mae_sd": float(np.std([r.rer_mae for r in fold_reports])),
        "rer_rmse": float(np.mean([r.rer_rmse for r in fold_reports])),
        "rer_bias": float(np.mean([r.rer_bias for r in fold_reports])),
        "vo2_mae_mlmin": float(np.mean([r.vo2_mae_mlmin for r in fold_reports])),
        "vo2_bias_mlmin": float(np.mean([r.vo2_bias_mlmin for r in fold_reports])),
        "lt2_auc": float(np.nanmean([r.lt2_auc for r in fold_reports])),
        "lt2_brier": float(np.mean([r.lt2_brier for r in fold_reports])),
        "rer_mae_by_intensity": _aggregate_bin_mae(fold_reports),
    }


def _aggregate_bin_mae(reports: list[MultitaskReport]) -> list[tuple[str, int, float]]:
    n_bins = len(reports[0].rer_mae_by_intensity)
    out = []
    for bin_i in range(n_bins):
        label = reports[0].rer_mae_by_intensity[bin_i][0]
        maes = [r.rer_mae_by_intensity[bin_i][2] for r in reports]
        ns = [r.rer_mae_by_intensity[bin_i][1] for r in reports]
        out.append((label, int(np.mean(ns)), float(np.nanmean(maes))))
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
        help="If set, also train on all data and save the model to this directory.",
    )
    args = p.parse_args(argv)

    print(f"LightGBM multitask CV: {args.data}")
    print(f"  {args.k}-fold subject-wise, {args.n_estimators} boosting rounds")
    print()
    res = cross_validate(
        args.data, k=args.k, seed=args.seed, n_estimators=args.n_estimators
    )
    print()
    print(f"=== LightGBM multitask result ===")
    print(f"  RER MAE     : {res['rer_mae']:.4f} ± {res['rer_mae_sd']:.4f}")
    print(f"  RER RMSE    : {res['rer_rmse']:.4f}")
    print(f"  RER bias    : {res['rer_bias']:+.4f}")
    print(f"  VO2 MAE     : {res['vo2_mae_mlmin']:.0f} mL/min")
    print(f"  VO2 bias    : {res['vo2_bias_mlmin']:+.0f} mL/min")
    print(f"  LT2 AUC     : {res['lt2_auc']:.3f}")
    print(f"  LT2 Brier   : {res['lt2_brier']:.4f}")
    print(f"  RER MAE by intensity bin:")
    for label, n, mae in res["rer_mae_by_intensity"]:
        if n > 0:
            print(f"    {label:>10s}  n_avg={n:>6d}  MAE={mae:.4f}")

    if args.save_model is not None:
        print()
        print(f"Training final model on all data and saving to {args.save_model}")
        ds = load_dataset(args.data)
        td = build_training_data(ds.samples, ds.subjects, ds.sessions)
        final_model = fit_multitask(td, n_estimators=args.n_estimators)
        final_model.save(args.save_model)
        print(f"Saved.")


if __name__ == "__main__":
    main()
