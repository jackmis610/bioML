"""Top-level CLI dispatch.

Usage:
  python -m bioml generate   # synthetic data generator -> parquet
  python -m bioml train      # LightGBM multitask CV, optionally save a model
  python -m bioml evaluate   # full report: baselines, calibration, SHAP, ablations
  python -m bioml serve      # FastAPI inference service
"""

from __future__ import annotations

import sys

COMMANDS = ("generate", "train", "evaluate", "serve")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        if len(sys.argv) >= 2:
            print(f"Unknown command: {sys.argv[1]}")
        print(__doc__)
        sys.exit(1)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "generate":
        from bioml.generator.run import main as run_main

        run_main(rest)
    elif cmd == "train":
        from bioml.train import main as train_main

        train_main(rest)
    elif cmd == "evaluate":
        from bioml.eval.report import main as report_main

        report_main(rest)
    elif cmd == "serve":
        from bioml.serve.app import main as serve_main

        serve_main(rest)


if __name__ == "__main__":
    main()
