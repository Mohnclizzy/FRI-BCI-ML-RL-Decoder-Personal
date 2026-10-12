"""Verify the committed PhysioNet subset without third-party dependencies."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "data" / "physionet"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    expected = {(subject, run) for subject in manifest["included_subjects"]
                for run in manifest["included_runs"]}
    actual = {(record["subject"], record["run"]) for record in files}
    if len(files) != manifest["file_count"] or actual != expected or len(actual) != len(files):
        raise SystemExit("Manifest does not list exactly the expected subject/run subset.")
    if len({record["path"] for record in files}) != len(files):
        raise SystemExit("Manifest contains duplicate file paths.")
    total = 0
    for record in files:
        path = (root / manifest["recording_root"] / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise SystemExit(f"Recording path escapes the dataset directory: {record['path']}")
        if not path.is_file():
            raise SystemExit(f"Missing recording: {record['path']}")
        size = path.stat().st_size
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if size != record["size_bytes"] or digest != record["sha256"]:
            raise SystemExit(f"Size or SHA-256 mismatch: {record['path']}")
        total += size
        print(f"OK {record['path']}")
    if total != manifest["total_bytes"]:
        raise SystemExit("Total recording size does not match the manifest.")
    print(f"Verified {len(files)} original EDF recordings ({total:,} bytes).")


if __name__ == "__main__":
    main()
