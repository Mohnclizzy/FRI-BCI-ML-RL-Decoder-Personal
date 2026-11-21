"""Calibrate per-person CSP/LDA models for S002 and S003.

Workaround for an MNE 1.13.2 bug: mne.datasets.eegbci.load_data crashes with
  AttributeError: 'PosixPath' object has no attribute 'split'
inside its internal path-update logic (_do_path_update). The EEGMMIDB data is
already on disk, so we monkeypatch eegbci.load_data to resolve the EDF paths
directly from disk (no download, no MNE path-update code path).

Pipeline per subject (mirrors the S001R12 protocol):
  1. baseline.prepare(config)  -> windows.npz / metadata.json / prepared_config.json
  2. compare_subjects.compare(config, dest) -> S{S}_model.joblib (calibration_subject set)
     train on runs 4/8 (held_out_run_same_subjects), test on run 12.
"""
import json
import shutil
import sys
from pathlib import Path

# Make the project root importable regardless of the script's own directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- 1. Monkeypatch eegbci.load_data to resolve from disk -------------------
import mne.datasets.eegbci as _eegbci


def _load_data_disk(subjects, runs, *, path=None, **kw):
    if isinstance(subjects, int):
        subjects = [subjects]
    if isinstance(runs, int):
        runs = [runs]
    base = Path(path).expanduser() / "MNE-eegbci-data/files/eegmmidb/1.0.0"
    out = []
    for s in subjects:
        for r in runs:
            p = base / f"S{s:03d}" / f"S{s:03d}R{r:02d}.edf"
            if not p.exists():
                raise FileNotFoundError(f"Missing EDF: {p}")
            out.append(str(p))
    return out


_eegbci.load_data = _load_data_disk

# --- 2. Import the project modules (they bind eegbci.load_data at call time) --
from eeg_decoder import baseline  # noqa: E402
from eeg_decoder.compare_subjects import compare  # noqa: E402
import joblib  # noqa: E402

# sanity: confirm the patch actually takes effect through baseline's binding
assert baseline.eegbci.load_data is _load_data_disk, "monkeypatch did not propagate"

# --- 3. Per-subject prepare + compare ---------------------------------------
ROOT = Path("results/robomimic_eeg_s002s003_2026-11-06")
BASE_CFG = json.loads(Path("configs/baseline.json").read_text(encoding="utf-8"))

results = {}
for S in (2, 3):
    cfg = dict(BASE_CFG)
    cfg["subjects"] = [S]
    cfg["runs"] = [4, 8, 12]
    cfg["test_run"] = 12
    cfg["test_subjects"] = []
    cfg["data_dir"] = "data/physionet"
    cfg["output_dir"] = str(ROOT / f"S{S:03d}_prepared")

    # fresh dirs
    prep_dir = Path(cfg["output_dir"])
    if prep_dir.exists():
        shutil.rmtree(prep_dir)
    model_dir = ROOT / f"S{S:03d}_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)

    print(f"\n===== S{S:03d}: prepare =====", flush=True)
    baseline.prepare(cfg)

    print(f"===== S{S:03d}: compare (train 4/8, test 12) =====", flush=True)
    report = compare(cfg, model_dir, "csp")

    model_path = model_dir / f"S{S:03d}_model.joblib"
    art = joblib.load(model_path)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    results[f"S{S:03d}"] = {
        "model": str(model_path),
        "calibration_subject": art.get("calibration_subject"),
        "model_family": art.get("model_family"),
        "pooled": metrics.get("pooled"),
        "individual": metrics.get("individual"),
    }
    print(f"===== S{S:03d}: metrics =====", flush=True)
    print(json.dumps({"pooled": metrics["pooled"], "individual": metrics["individual"]}, indent=2))
    print(f"calibration_subject={art.get('calibration_subject')}  "
          f"model_family={art.get('model_family')}  "
          f"model={model_path}", flush=True)

(ROOT / "calibration_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print("\nWROTE", ROOT / "calibration_summary.json", flush=True)
print("DONE", flush=True)
