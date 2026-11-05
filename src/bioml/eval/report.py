"""Reproducible evaluation report: the command behind every number in the README.

``python -m bioml evaluate`` runs the LightGBM model and both baselines through
identical subject-wise folds, then adds what a single MAE hides:

* substrate-regime metrics (fat/CHO error inside the Frayn-valid envelope and
  precision/recall of the inference gate),
* LT2 calibration (reliability table, ECE),
* error and bias by intensity bin,
* feature attribution (gain importance + out-of-fold TreeSHAP),
* feature-group ablations ("no power meter", "no CGM", "no session clock", ...).

Outputs ``metrics.json`` (machine-readable) and ``REPORT.md`` (human-readable).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bioml import __version__
from bioml.baselines.hr_zone import HRZoneBaseline
from bioml.baselines.population_curve import PopulationCurveBaseline
from bioml.eval.features import (
    FEATURE_COLS,
    FEATURE_GROUPS,
    build_training_data,
    feature_cols_excluding,
    group_of,
)
from bioml.eval.loaders import load_dataset
from bioml.eval.metrics import (
    SubstrateReport,
    evaluate_substrate,
    expected_calibration_error,
    reliability_table,
)
from bioml.inference import DEFAULT_POLICY, emit_substrate
from bioml.train import CVRun, LightGBMMultitask, _git_sha, fit_multitask, run_cv, summarize_cv

ABLATIONS: dict[str, tuple[str, ...]] = {
    "no_time": ("time",),
    "no_hr_dynamics": ("hr_dynamics",),
    "no_power": ("power",),
    "no_cgm": ("cgm",),
    "watch_only": ("power", "cgm"),
    "hr_only_minimal": ("power", "cgm", "time", "hr_dynamics", "environment"),
}


def substrate_from_oof(oof: pl.DataFrame) -> SubstrateReport:
    emitted = emit_substrate(
        oof["rer_pred"].to_numpy(),
        oof["vo2_pred_mlmin"].to_numpy(),
        oof["lt2_prob"].to_numpy(),
        DEFAULT_POLICY,
    )
    return evaluate_substrate(
        oof["fat_ox_g_per_min"].to_numpy(),
        oof["cho_ox_g_per_min"].to_numpy(),
        emitted["fat_ox_pred_g_per_min"],
        emitted["cho_ox_pred_g_per_min"],
        emitted["substrate_valid"],
    )


def _pooled_calibration(oof: pl.DataFrame) -> dict[str, Any]:
    y = oof["above_lt2_flag"].to_numpy()
    p = oof["lt2_prob"].to_numpy()
    return {
        "ece": expected_calibration_error(y, p),
        "reliability": reliability_table(y, p),
    }


def _fmt(x: float, nd: int = 3) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _pct_delta(new: float, ref: float) -> str:
    if ref == 0 or np.isnan(ref) or np.isnan(new):
        return "n/a"
    return f"{(new - ref) / abs(ref) * 100:+.1f}%"


def run_report(
    data: Path,
    out: Path,
    k: int = 5,
    seed: int = 42,
    n_estimators: int = 400,
    run_ablations: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    ds = load_dataset(data)
    n_subjects = ds.subjects.height
    n_rows = int(ds.samples.select(pl.len()).collect().item())

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"Evaluating on {data}: {n_subjects} subjects, {n_rows:,} rows, {k}-fold subject-wise")

    log("[1/4] models + baselines on identical folds")
    runs: dict[str, CVRun] = {}
    runs["hr_zone"] = run_cv(ds, HRZoneBaseline, "hr_zone", k=k, seed=seed, verbose=verbose)
    runs["population_curve"] = run_cv(
        ds, PopulationCurveBaseline, "population_curve", k=k, seed=seed, verbose=verbose
    )
    runs["lightgbm"] = run_cv(
        ds,
        lambda: LightGBMMultitask(n_estimators=n_estimators),
        "lightgbm",
        k=k,
        seed=seed,
        compute_shap=True,
        verbose=verbose,
    )

    models: dict[str, Any] = {}
    for name, run in runs.items():
        summary = summarize_cv(run)
        summary["substrate"] = substrate_from_oof(run.oof).to_dict()
        summary["calibration"] = _pooled_calibration(run.oof)
        models[name] = summary

    log("[2/4] feature attribution")
    full_td = build_training_data(ds.samples, ds.subjects, ds.sessions)
    final_model = fit_multitask(full_td, n_estimators=n_estimators)
    attribution = {
        "gain_importance": final_model.feature_importance("gain"),
        "shap_mean_abs_oof": runs["lightgbm"].shap_mean_abs,
    }

    ablations: dict[str, Any] = {}
    if run_ablations:
        log(f"[3/4] ablations ({len(ABLATIONS)} configs)")
        for name, groups in ABLATIONS.items():
            cols = feature_cols_excluding(groups)
            log(f"  - {name}: drop {list(groups)} -> {len(cols)} features")
            run = run_cv(
                ds,
                lambda: LightGBMMultitask(n_estimators=n_estimators),
                name,
                k=k,
                seed=seed,
                feature_cols=cols,
                verbose=False,
            )
            s = summarize_cv(run)
            s["dropped_groups"] = list(groups)
            s["n_features"] = len(cols)
            s["substrate"] = substrate_from_oof(run.oof).to_dict()
            ablations[name] = s
    else:
        log("[3/4] ablations skipped")

    log("[4/4] writing report")
    result: dict[str, Any] = {
        "provenance": {
            "data": str(data),
            "n_subjects": n_subjects,
            "n_rows": n_rows,
            "k": k,
            "seed": seed,
            "n_estimators": n_estimators,
            "n_features": len(FEATURE_COLS),
            "feature_groups": {g: list(c) for g, c in FEATURE_GROUPS.items()},
            "bioml_version": __version__,
            "git_sha": _git_sha(),
            "generated_at": datetime.now(UTC).isoformat(),
            "wall_seconds": None,
        },
        "models": models,
        "attribution": attribution,
        "ablations": ablations,
    }
    result["provenance"]["wall_seconds"] = round(time.perf_counter() - t_start, 1)

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "REPORT.md").write_text(render_markdown(result))
    log(f"Wrote {out / 'metrics.json'} and {out / 'REPORT.md'}")
    return result


def render_markdown(r: dict[str, Any]) -> str:
    prov = r["provenance"]
    m = r["models"]
    lgb, pop, hz = m["lightgbm"], m["population_curve"], m["hr_zone"]
    lines: list[str] = []
    a = lines.append

    a("# bioML evaluation report")
    a("")
    a(
        f"Generated {prov['generated_at'][:19]}Z from `{prov['data']}` "
        f"({prov['n_subjects']} subjects, {prov['n_rows']:,} rows). "
        f"{prov['k']}-fold subject-wise CV, seed {prov['seed']}, "
        f"{prov['n_estimators']} boosting rounds, {prov['n_features']} features. "
        f"bioml {prov['bioml_version']} @ `{prov['git_sha']}`. "
        f"Wall time {prov['wall_seconds']}s."
    )
    a("")
    a("## Headline (same folds for every row)")
    a("")
    a("| Model | RER MAE | VO2 MAE (mL/min) | VO2 bias | LT2 AUC | LT2 Brier | LT2 ECE |")
    a("|---|---|---|---|---|---|---|")
    for label, s in (
        ("HR-zone heuristic", hz),
        ("Population curve", pop),
        ("**LightGBM multitask**", lgb),
    ):
        a(
            f"| {label} | {_fmt(s['rer_mae'], 4)} ± {_fmt(s['rer_mae_sd'], 4)} "
            f"| {s['vo2_mae_mlmin']:.0f} ± {s['vo2_mae_mlmin_sd']:.0f} "
            f"| {s['vo2_bias_mlmin']:+.0f} "
            f"| {_fmt(s['lt2_auc'])} | {_fmt(s['lt2_brier'], 4)} | {_fmt(s['lt2_ece'], 4)} |"
        )
    a("")
    a(
        f"LightGBM vs population curve: RER MAE {_pct_delta(lgb['rer_mae'], pop['rer_mae'])}, "
        f"VO2 MAE {_pct_delta(lgb['vo2_mae_mlmin'], pop['vo2_mae_mlmin'])}, "
        f"LT2 Brier {_pct_delta(lgb['lt2_brier'], pop['lt2_brier'])}. "
        f"vs HR-zone: RER MAE {_pct_delta(lgb['rer_mae'], hz['rer_mae'])}, "
        f"VO2 MAE {_pct_delta(lgb['vo2_mae_mlmin'], hz['vo2_mae_mlmin'])}."
    )
    a("")
    a("## Substrate emission (what the product actually outputs)")
    a("")
    a(
        "Fat/CHO oxidation is computed with Frayn only where the gate "
        "(`rer_pred < 0.99 AND P[LT2] < 0.5`) says it is valid. Truth validity is the generator's "
        "own Frayn envelope. Errors are over rows where both agree."
    )
    a("")
    a(
        "| Model | fat-ox MAE (g/min) | fat-ox bias | CHO-ox MAE (g/min) "
        "| gate precision | gate recall | gate accuracy | rows scored |"
    )
    a("|---|---|---|---|---|---|---|---|")
    for label, s in (
        ("HR-zone heuristic", hz),
        ("Population curve", pop),
        ("**LightGBM multitask**", lgb),
    ):
        sub = s["substrate"]
        a(
            f"| {label} | {_fmt(sub['fat_ox_mae_g_per_min'], 3)} "
            f"| {sub['fat_ox_bias_g_per_min']:+.3f} "
            f"| {_fmt(sub['cho_ox_mae_g_per_min'], 3)} | {_fmt(sub['gate_precision'])} "
            f"| {_fmt(sub['gate_recall'])} | {_fmt(sub['gate_accuracy'])} "
            f"| {sub['n_both_valid']:,} |"
        )
    a("")
    a("## Error by intensity (LightGBM)")
    a("")
    a("| %VO2max bin | n/fold | RER MAE | RER bias | VO2 MAE | VO2 bias |")
    a("|---|---|---|---|---|---|")
    for (label, n, rmae), (_, _, rb), (_, _, vmae), (_, _, vb) in zip(
        lgb["rer_mae_by_intensity"],
        lgb["rer_bias_by_intensity"],
        lgb["vo2_mae_by_intensity"],
        lgb["vo2_bias_by_intensity"],
        strict=True,
    ):
        if n:
            a(f"| {label} | {n:,} | {rmae:.4f} | {rb:+.4f} | {vmae:.0f} | {vb:+.0f} |")
    a("")
    a("## LT2 head calibration (LightGBM, pooled out-of-fold)")
    a("")
    a(f"ECE = {_fmt(lgb['calibration']['ece'], 4)}")
    a("")
    a("| P(LT2) bin | n | mean predicted | observed rate |")
    a("|---|---|---|---|")
    for lo, hi, n, pred, obs in lgb["calibration"]["reliability"]:
        if n:
            a(f"| {lo:.1f}-{hi:.1f} | {n:,} | {pred:.3f} | {obs:.3f} |")
    a("")
    a("## Feature attribution")
    a("")
    shap = r["attribution"]["shap_mean_abs_oof"] or {}
    gain = r["attribution"]["gain_importance"]
    for head, title in (("rer", "RER head"), ("vo2", "VO2 head"), ("lt2", "LT2 head")):
        a(f"### {title}: top features by out-of-fold mean |SHAP|")
        a("")
        a("| Feature | Group | mean abs SHAP | gain share |")
        a("|---|---|---|---|")
        ranked = sorted(shap.get(head, {}).items(), key=lambda kv: -kv[1])[:10]
        for f, v in ranked:
            a(f"| `{f}` | {group_of(f)} | {v:.4g} | {gain[head].get(f, 0.0) * 100:.1f}% |")
        a("")
    if r["ablations"]:
        a("## Feature-group ablations (full subject-wise CV per row)")
        a("")
        a("| Config | dropped | #feat | RER MAE | Δ | VO2 MAE | Δ | LT2 AUC | fat-ox MAE |")
        a("|---|---|---|---|---|---|---|---|---|")
        a(
            f"| full | – | {prov['n_features']} | {lgb['rer_mae']:.4f} | – "
            f"| {lgb['vo2_mae_mlmin']:.0f} | – | {lgb['lt2_auc']:.3f} "
            f"| {_fmt(lgb['substrate']['fat_ox_mae_g_per_min'], 3)} |"
        )
        for name, s in r["ablations"].items():
            a(
                f"| {name} | {', '.join(s['dropped_groups'])} | {s['n_features']} "
                f"| {s['rer_mae']:.4f} | {_pct_delta(s['rer_mae'], lgb['rer_mae'])} "
                f"| {s['vo2_mae_mlmin']:.0f} "
                f"| {_pct_delta(s['vo2_mae_mlmin'], lgb['vo2_mae_mlmin'])} "
                f"| {s['lt2_auc']:.3f} | {_fmt(s['substrate']['fat_ox_mae_g_per_min'], 3)} |"
            )
        a("")
    a("## Reproduce")
    a("")
    a("```bash")
    a(
        f"uv run python -m bioml generate --n-subjects {prov['n_subjects']} "
        f"--seed {prov['seed']} --out {prov['data']}"
    )
    a(
        f"uv run python -m bioml evaluate --data {prov['data']} --k {prov['k']} "
        f"--seed {prov['seed']} --n-estimators {prov['n_estimators']} --out reports/v2"
    )
    a("```")
    a("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bioml.evaluate")
    p.add_argument("--data", type=Path, default=Path("data/synthetic/v1"))
    p.add_argument("--out", type=Path, default=Path("reports/v2"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--skip-ablations", action="store_true")
    args = p.parse_args(argv)
    run_report(
        args.data,
        args.out,
        k=args.k,
        seed=args.seed,
        n_estimators=args.n_estimators,
        run_ablations=not args.skip_ablations,
    )


if __name__ == "__main__":
    main()
