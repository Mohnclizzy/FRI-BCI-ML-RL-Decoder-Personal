"""Hash the corrected EEG batch, its input traces, and EDFs for provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/isaac/eeg-matched-5p-continuation-20261103T0300Z-b120s"
OUT = ROOT / "results/corrected_eeg_2026-11-03-complete/provenance_sha256.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    cases = list(csv.DictReader((RUN / "results.tsv").open(encoding="utf-8"), delimiter="\t"))
    if len(cases) != 18 or any(row["exit_code"] != "0" for row in cases):
        raise ValueError("Expected all 18 corrected cases to have exit code zero")
    artifacts = [RUN / "manifest.txt", RUN / "results.tsv"]
    for row in cases:
        telemetry = RUN / row["condition"] / "telemetry.jsonl"
        if not telemetry.is_file() or telemetry.stat().st_size == 0:
            raise FileNotFoundError(telemetry)
        case_dir = telemetry.parent
        artifacts.extend((telemetry, case_dir / "run.log"))
        initialization = case_dir / "initialization.jsonl"
        if initialization.exists():
            artifacts.append(initialization)
    for person in ("S006", "S007", "S008", "S009", "S010"):
        artifacts.append(ROOT / f"outputs/isaac/eeg-batch-inputs/{person}R12-decisions.jsonl")
        artifacts.append(ROOT / f"data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/{person}/{person}R12.edf")
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing evidence: " + ", ".join(missing))
    report = {
        "scope": "Completed corrected exploratory EEG scripted batch; SHA256 of run, inputs and EDF files",
        "run_directory": str(RUN.relative_to(ROOT)),
        "case_count_including_controls": len(cases),
        "all_exit_codes_zero": True,
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in artifacts},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Hashed {len(artifacts)} files: {OUT}")


if __name__ == "__main__":
    main()
