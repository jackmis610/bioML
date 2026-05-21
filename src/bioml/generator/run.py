"""End-to-end generator: subject -> session -> physiology -> sensors -> parquet."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bioml.config import load_spec
from bioml.generator.physiology import run_session
from bioml.generator.sensors import apply_sensor_overlay
from bioml.generator.sessions import sample_session
from bioml.generator.subjects import sample_subject
from bioml.schemas import Session, Subject


def build_samples_df(
    subject: Subject,
    session: Session,
    truth: dict[str, Any],
    obs: dict[str, Any],
) -> pl.DataFrame:
    n = len(truth["t_s"])
    is_warmup = np.zeros(n, dtype=bool)
    is_warmup[:180] = True
    is_recovery = np.zeros(n, dtype=bool)
    is_recovery[-180:] = True

    data: dict[str, Any] = {
        "session_id": [session.session_id] * n,
        "subject_id": [subject.subject_id] * n,
        "t_s": truth["t_s"].astype(np.int64),
        "protocol_target_power_w": truth["power_lab_w"],
        "power_lab_w": truth["power_lab_w"],
        "vo2_lab_mlmin": truth["vo2_lab_mlmin"],
        "vco2_lab_mlmin": truth["vco2_lab_mlmin"],
        "rer_lab": truth["rer_lab"],
        "hr_lab_bpm": truth["hr_lab_bpm"],
        "pct_vo2max_lab": truth["pct_vo2max_lab"],
        "ve_lab_lmin": np.full(n, np.nan),
        "fat_ox_g_per_min": truth["fat_ox_g_per_min"],
        "cho_ox_g_per_min": truth["cho_ox_g_per_min"],
        "above_lt2_flag": truth["above_lt2_flag"],
        "muscle_glycogen_mmol_per_kg_true": truth["muscle_glycogen_mmol_per_kg_true"],
        "blood_glucose_mg_dl": np.full(n, np.nan),
        "blood_lactate_mmol_l": np.full(n, np.nan),
        "hr_wearable_bpm": obs["hr_wearable_bpm"],
        "power_wearable_w": obs["power_wearable_w"],
        "cadence_rpm": obs["cadence_rpm"],
        "cgm_glucose_mg_dl": obs["cgm_glucose_mg_dl"],
        "accelerometer_movement_index": obs["accelerometer_movement_index"],
        "is_warmup": is_warmup,
        "is_recovery": is_recovery,
    }
    df = pl.DataFrame(data)
    nullable_float_cols = [
        "fat_ox_g_per_min",
        "cho_ox_g_per_min",
        "ve_lab_lmin",
        "blood_glucose_mg_dl",
        "blood_lactate_mmol_l",
        "hr_wearable_bpm",
        "power_wearable_w",
        "cgm_glucose_mg_dl",
        "accelerometer_movement_index",
    ]
    df = df.with_columns(
        [
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in nullable_float_cols
            if c in df.columns
        ]
    )
    return df


def _models_to_df(models: list[Any]) -> pl.DataFrame:
    return pl.DataFrame([m.model_dump(mode="json") for m in models])


def generate_dataset(
    n_subjects: int,
    out_dir: Path,
    seed: int = 42,
    log_every: int = 100,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    spec = load_spec()

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    subjects: list[Subject] = []
    sessions: list[Session] = []
    total_rows = 0
    t0 = time.time()

    for i in range(n_subjects):
        s = sample_subject(spec, rng)
        session, power = sample_session(s, rng)
        truth = run_session(s, session, power, spec, rng)
        obs = apply_sensor_overlay(truth, session, rng)

        samples_df = build_samples_df(s, session, truth, obs)
        partition_dir = samples_dir / f"subject_id={s.subject_id}"
        partition_dir.mkdir(exist_ok=True)
        samples_df.write_parquet(partition_dir / f"{session.session_id}.parquet")

        subjects.append(s)
        sessions.append(session)
        total_rows += len(samples_df)

        if log_every and (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"  {i + 1}/{n_subjects} subjects | {total_rows:,} sample rows | "
                f"{rate:.1f} subj/s | ETA {(n_subjects - i - 1) / rate:.0f}s"
            )

    _models_to_df(subjects).write_parquet(out_dir / "subjects.parquet")
    _models_to_df(sessions).write_parquet(out_dir / "sessions.parquet")

    elapsed = time.time() - t0
    print(
        f"Done. {n_subjects} subjects, {total_rows:,} sample rows in {elapsed:.1f}s "
        f"-> {out_dir.resolve()}"
    )
    return {
        "n_subjects": n_subjects,
        "n_sessions": len(sessions),
        "n_rows": total_rows,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bioml.generator")
    p.add_argument("--n-subjects", type=int, default=100)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic/v1"),
        help="output directory; will be created",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args(argv)
    generate_dataset(
        n_subjects=args.n_subjects,
        out_dir=args.out,
        seed=args.seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
