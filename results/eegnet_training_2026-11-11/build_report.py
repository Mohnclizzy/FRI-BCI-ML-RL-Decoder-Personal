"""Idempotent wrapper that runs the unified November 11 report and manifest builder.

Preserves all 155 artifact hashes in `artifact_hashes.json` and regenerates both
`summary.md` and `evaluation_summary.md` without opening EEG recordings or
dropping evaluation/pretraining hashes.
"""
from __future__ import annotations

from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "build_evaluation_report.py"
    runpy.run_path(str(target), run_name="__main__")
