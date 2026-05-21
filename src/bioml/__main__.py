"""Top-level CLI dispatch.

Usage:
  python -m bioml generate   # synthetic data generator
  python -m bioml train      # LightGBM multitask trainer
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "generate":
        from bioml.generator.run import main as run_main

        run_main(rest)
    elif cmd == "train":
        from bioml.train import main as train_main

        train_main(rest)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
