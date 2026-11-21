"""Audit cue-schedule predictability and variable cue durations on BCI Competition IV Dataset 1.

Reads official 1000 Hz ground-truth labels (`true_labels.zip`), verifies its SHA-256 digest,
computes full-cue duration statistics (including each 1.0 s post-cue mask) across all 7
subjects (`ds1a`–`ds1g`), and evaluates clock-only baselines (Run 1 continuous-cue replay on
Runs 2–4 and a fixed 4.1 s alternating timer) on 4 Hz endpoints.

Usage:
    python -m analysis.audit_bciciv_ds1_clock --labels-zip <path/to/true_labels.zip> --out report/final/bciciv_ds1_clock_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

import numpy as np
import scipy.io

ALL_SUBJECTS = ("a", "b", "c", "d", "e", "f", "g")
HUMAN_SUBJECTS = ("a", "b", "f", "g")
ARTIFICIAL_SUBJECTS = ("c", "d", "e")
EXPECTED_LABELS_ZIP_SHA256 = "b34de8373d6476ff534f3a7496397ed86726c04dd12f610d84b8baaa71e28c72"


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_4_runs_1000hz(y_1000: np.ndarray) -> list[tuple[int, int]]:
    """Split a 1000 Hz true_y stream into its 4 evaluation runs via inter-run NaN blocks (>2 s)."""
    isnan = np.isnan(y_1000)
    changes = np.flatnonzero(np.diff(isnan.astype(int)) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(y_1000)]))
    long_nans = [(int(s), int(e)) for s, e in zip(starts, stops) if isnan[s] and (e - s) > 2000]
    if len(long_nans) != 5:
        raise ValueError(f"Expected 5 inter-run NaN blocks (>2 s), found {len(long_nans)}")
    return [(long_nans[i][1] - 1000, long_nans[i + 1][0]) for i in range(4)]


def continuous_cue_labels(y_run_1000: np.ndarray) -> tuple[np.ndarray, list[float], list[float]]:
    """Propagate each cued class backward across its 1.0 s post-cue mask and collect full cue durations."""
    valid = np.where(np.isnan(y_run_1000), -99, np.where(y_run_1000 == 0, 0, 1))
    changes = np.flatnonzero(np.diff(valid) != 0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(valid)]))
    cont = np.full(len(y_run_1000), -1, dtype=int)
    rest_durs: list[float] = []
    mi_durs: list[float] = []
    for s, e in zip(starts, stops):
        lbl = int(valid[s])
        if lbl in (0, 1):
            cue_start = max(0, int(s) - 1000)
            cont[cue_start:int(e)] = lbl
            dur_s = (int(e) - cue_start) / 1000.0
            if lbl == 0:
                rest_durs.append(dur_s)
            else:
                mi_durs.append(dur_s)
    return cont, rest_durs, mi_durs


def balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary balanced accuracy over classes 0 and 1."""
    r0 = float(np.mean(y_pred[y_true == 0] == 0))
    r1 = float(np.mean(y_pred[y_true == 1] == 1))
    return 0.5 * (r0 + r1)


def _duration_stats(vals: list[float]) -> dict[str, float]:
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "sd": round(statistics.stdev(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def audit_clock_baselines(labels_zip_path: Path) -> dict:
    """Compute cue duration summaries and clock-only baselines across all 7 BCICIV-1 subjects."""
    actual_sha = sha256_file(labels_zip_path)
    if actual_sha != EXPECTED_LABELS_ZIP_SHA256:
        raise ValueError(f"true_labels.zip SHA-256 mismatch: {actual_sha}")

    lbl_zip = zipfile.ZipFile(labels_zip_path)
    participants: dict[str, dict] = {}

    for sub in ALL_SUBJECTS:
        sub_key = f"ds1{sub}"
        member = f"BCICIV_eval_{sub_key}_1000Hz_true_y.mat"
        raw_bytes = lbl_zip.read(member)
        y_1000 = scipy.io.loadmat(io.BytesIO(raw_bytes), squeeze_me=True)["true_y"].astype(float)
        runs = split_4_runs_1000hz(y_1000)

        # Global 4 Hz endpoints for fixed 4.1 s timer baseline across all annotated samples
        ep_global = np.arange(0, len(y_1000), 250)
        y_ep = y_1000[ep_global]
        ann = ~np.isnan(y_ep)
        y_true_global = np.where(y_ep[ann] == 0, 0, 1)
        t_ep = ep_global[ann] / 1000.0
        pred_4_1s = (np.rint(t_ep * 1000).astype(int) // 4100) % 2
        ba_4_1s = balanced_accuracy_binary(y_true_global, pred_4_1s)

        # Per-run continuous cue arrays & Run 1 -> Runs 2-4 continuous replay
        run_cont = []
        rest_durs_all: list[float] = []
        mi_durs_all: list[float] = []
        for s_idx, e_idx in runs:
            yr = y_1000[s_idx:e_idx]
            cont, r_d, m_d = continuous_cue_labels(yr)
            run_cont.append((yr, cont))
            rest_durs_all.extend(r_d)
            mi_durs_all.extend(m_d)

        r1_cont = run_cont[0][1]
        per_run_ba: list[float] = []
        for k in range(1, 4):
            yr_k, _ = run_cont[k]
            pts = np.arange(0, len(yr_k), 250)
            pts_v = pts[~np.isnan(yr_k[pts])]
            yt_k = np.where(yr_k[pts_v] == 0, 0, 1)
            yp_k = np.array([r1_cont[p] if p < len(r1_cont) else 0 for p in pts_v], dtype=int)
            per_run_ba.append(round(balanced_accuracy_binary(yt_k, yp_k), 4))

        participants[sub_key] = {
            "subject_id": sub,
            "type": "human" if sub in HUMAN_SUBJECTS else "artificial",
            "mat_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "duration_s": round(len(y_1000) / 1000.0, 2),
            "endpoints_4hz_annotated": int(ann.sum()),
            "rest_cue_duration_s": _duration_stats(rest_durs_all),
            "mi_cue_duration_s": _duration_stats(mi_durs_all),
            "run1_to_runs234_continuous_cue_per_run": per_run_ba,
            "run1_to_runs234_continuous_cue_bal_acc": round(statistics.mean(per_run_ba), 4),
            "periodic_4_1s_bal_acc": round(ba_4_1s, 4),
        }

    human_keys = [f"ds1{s}" for s in HUMAN_SUBJECTS]
    art_keys = [f"ds1{s}" for s in ARTIFICIAL_SUBJECTS]
    human_rest_mins = [participants[k]["rest_cue_duration_s"]["min"] for k in human_keys]
    human_rest_maxs = [participants[k]["rest_cue_duration_s"]["max"] for k in human_keys]
    human_mi_mins = [participants[k]["mi_cue_duration_s"]["min"] for k in human_keys]
    human_mi_maxs = [participants[k]["mi_cue_duration_s"]["max"] for k in human_keys]

    return {
        "dataset": "BCI Competition IV Dataset 1",
        "true_labels_zip_sha256": EXPECTED_LABELS_ZIP_SHA256,
        "human_subjects": human_keys,
        "artificial_subjects_excluded": art_keys,
        "human_summary": {
            "n_human_participants": len(human_keys),
            "total_human_mi_episodes": sum(participants[k]["mi_cue_duration_s"]["count"] for k in human_keys),
            "total_human_rest_episodes": sum(participants[k]["rest_cue_duration_s"]["count"] for k in human_keys),
            "rest_cue_duration_range_s": [min(human_rest_mins), max(human_rest_maxs)],
            "mi_cue_duration_range_s": [min(human_mi_mins), max(human_mi_maxs)],
            "mean_run1_to_runs234_continuous_cue_bal_acc": round(
                statistics.mean(participants[k]["run1_to_runs234_continuous_cue_bal_acc"] for k in human_keys), 4
            ),
            "mean_periodic_4_1s_bal_acc": round(
                statistics.mean(participants[k]["periodic_4_1s_bal_acc"] for k in human_keys), 4
            ),
        },
        "artificial_schedule_note": {
            "ds1c_run1_vs_run3_exact_match": participants["ds1c"]["run1_to_runs234_continuous_cue_per_run"][1] == 1.0,
            "ds1c_per_run_bal_acc": participants["ds1c"]["run1_to_runs234_continuous_cue_per_run"],
        },
        "participants": participants,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-zip", type=Path, required=True, help="Path to official true_labels.zip")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSON file.")
    args = parser.parse_args(argv)

    result = audit_clock_baselines(args.labels_zip.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote clock audit: {args.out}")
    print("Human summary:", json.dumps(result["human_summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
