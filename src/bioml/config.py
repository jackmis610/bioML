"""Loader for research/spec.yaml — the operational physiology + sensor config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "research" / "spec.yaml"


def load_spec(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_SPEC_PATH
    with p.open("r") as f:
        return yaml.safe_load(f)


def stratum_value(
    block: dict[str, Any],
    stratum: str,
    sex: str | None = None,
) -> Any:
    """Resolve a per-stratum-(per-sex)-optional value.

    Spec blocks look like:
        by_stratum:
          untrained:    { M: [a, b], F: [c, d] }
          recreational: { all: [e, f] }
          well_trained: [g, h]              # not sex-split

    Returns the most specific value: sex-split if both sex and split exist,
    'all' bucket if present, otherwise the bare value.
    """
    by = block.get("by_stratum", block)
    entry = by[stratum]
    if isinstance(entry, dict):
        if sex is not None and sex in entry:
            return entry[sex]
        if "all" in entry:
            return entry["all"]
    return entry
