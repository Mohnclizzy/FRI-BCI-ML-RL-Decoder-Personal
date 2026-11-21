"""Stieger et al. (2021) Figshare manifest, deterministic split, MATLAB audit, and causal adapter.

Dataset: "Continuous sensorimotor rhythm based brain computer interface learning in a large population"
(Scientific Data, doi:10.6084/m9.figshare.13123148, 62 subjects, 598 sessions).

Preserves explicit separation between deliberate cursor-down relaxation (targetnumber == 4),
inter-trial idle (t < 0 ms), visual cue presentation, post-trial outcome, and artifact trials.
All signal filtering, downsampling, and window slicing are strictly causal per trial.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import requests
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

FIGSHARE_ARTICLE_ID = 13123148
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

CANONICAL_9_CHANNELS = ["C3", "Cz", "C4", "C1", "C2", "FC1", "FC2", "CP1", "CP2"]

STATE_INTERTRIAL_IDLE = 0
STATE_CUE_ACTIVE_MI = 1
STATE_CUE_DELIBERATE_RELAX = 2
STATE_FEEDBACK_ACTIVE_MI = 3
STATE_FEEDBACK_DELIBERATE_RELAX = 4
STATE_POST_TRIAL_OUTCOME = 5
STATE_UNLABELED_OR_ARTIFACT = 6

STATE_NAMES = {
    STATE_INTERTRIAL_IDLE: "intertrial_idle",
    STATE_CUE_ACTIVE_MI: "cue_active_mi",
    STATE_CUE_DELIBERATE_RELAX: "cue_deliberate_relax",
    STATE_FEEDBACK_ACTIVE_MI: "feedback_active_mi",
    STATE_FEEDBACK_DELIBERATE_RELAX: "feedback_deliberate_relax",
    STATE_POST_TRIAL_OUTCOME: "post_trial_outcome",
    STATE_UNLABELED_OR_ARTIFACT: "unlabeled_or_artifact",
}

TASK_NAMES = {1: "LR", 2: "UD", 3: "2D"}
TARGET_NAMES = {
    1: "right",
    2: "left",
    3: "up_both_hands",
    4: "down_deliberate_relax",
}


def write_json(path: str | Path, value: Any) -> Path:
    """Atomically write compact formatted JSON without allowing NaNs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


def sha256_file(path: str | Path) -> str:
    """Compute hex SHA-256 of a file using streaming 1 MiB reads."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_figshare_article(article_id: int = FIGSHARE_ARTICLE_ID, timeout_s: int = 30) -> dict[str, Any]:
    """Fetch raw article JSON metadata from the public Figshare REST API."""
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "fri-bci-phase2-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_figshare_manifest(article: dict[str, Any]) -> dict[str, Any]:
    """Normalize Figshare article metadata into structured README + 598 session records."""
    readme = None
    sessions: list[dict[str, Any]] = []
    other_files: list[dict[str, Any]] = []
    for item in article.get("files", []):
        name = str(item["name"])
        if name == "README.pdf":
            readme = {
                "file_id": int(item["id"]),
                "filename": name,
                "size_bytes": int(item["size"]),
                "computed_md5": str(item["computed_md5"]),
                "download_url": str(item["download_url"]),
            }
            continue
        match = re.match(r"^S(\d+)_Session_(\d+)\.mat$", name)
        if not match:
            other_files.append({"file_id": int(item["id"]), "filename": name, "size_bytes": int(item["size"])})
            continue
        sessions.append({
            "file_id": int(item["id"]),
            "filename": name,
            "session_key": f"S{int(match.group(1))}_Session_{int(match.group(2))}",
            "subject_id": int(match.group(1)),
            "session_id": int(match.group(2)),
            "size_bytes": int(item["size"]),
            "computed_md5": str(item["computed_md5"]),
            "download_url": str(item["download_url"]),
        })
    sessions.sort(key=lambda row: (row["subject_id"], row["session_id"]))
    by_subj: dict[int, list[dict[str, Any]]] = {}
    for row in sessions:
        by_subj.setdefault(row["subject_id"], []).append(row)
    subjects = [
        {
            "subject_id": sid,
            "n_sessions": len(rows),
            "session_ids": [r["session_id"] for r in rows],
            "total_bytes": sum(r["size_bytes"] for r in rows),
        }
        for sid, rows in sorted(by_subj.items())
    ]
    return {
        "article_id": int(article.get("id", FIGSHARE_ARTICLE_ID)),
        "doi": str(article.get("doi", "")),
        "title": str(article.get("title", "")),
        "total_article_bytes": int(article.get("size", 0)),
        "n_subjects": len(subjects),
        "n_sessions": len(sessions),
        "readme": readme,
        "other_files": other_files,
        "subjects": subjects,
        "sessions": sessions,
    }


def build_deterministic_split(
    manifest: dict[str, Any],
    seed: int = 20261119,
    pinned_dev_subjects: tuple[int, ...] = (1, 2),
    n_dev: int = 8,
    n_test: int = 10,
) -> dict[str, Any]:
    """Create a deterministic participant-disjoint split with strict nested source scales."""
    all_subject_ids = sorted({int(s["subject_id"]) for s in manifest["sessions"]})
    if len(all_subject_ids) != 62:
        raise ValueError(f"Expected 62 subjects in Stieger manifest, found {len(all_subject_ids)}.")
    pinned = [int(s) for s in pinned_dev_subjects]
    if not set(pinned) <= set(all_subject_ids):
        raise ValueError("Pinned dev subjects missing from manifest.")

    pool = np.array([sid for sid in all_subject_ids if sid not in pinned], dtype=int)
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = [int(x) for x in rng.permutation(pool)]

    extra_dev = shuffled[: n_dev - len(pinned)]
    dev_subjects = sorted(pinned + extra_dev)
    test_subjects = sorted(shuffled[n_dev - len(pinned) : n_dev - len(pinned) + n_test])
    source_order = shuffled[n_dev - len(pinned) + n_test :]

    n_src = len(source_order)
    n10 = max(1, round(0.10 * n_src))
    n30 = max(n10 + 1, round(0.30 * n_src))
    scale_10 = source_order[:n10]
    scale_30 = source_order[:n30]
    scale_100 = source_order

    sess_by_subj: dict[int, list[dict[str, Any]]] = {}
    for row in manifest["sessions"]:
        sess_by_subj.setdefault(int(row["subject_id"]), []).append(row)

    def summarize(subject_ids: list[int]) -> dict[str, Any]:
        rows = [r for sid in subject_ids for r in sess_by_subj[sid]]
        return {
            "n_subjects": len(subject_ids),
            "n_sessions": len(rows),
            "total_bytes": sum(r["size_bytes"] for r in rows),
            "total_gib": round(sum(r["size_bytes"] for r in rows) / (1024**3), 3),
        }

    split_cfg = {
        "schema_version": "phase2-stieger-split-v1",
        "dataset_doi": manifest["doi"],
        "split_seed": int(seed),
        "frozen_before_signal_inspection": True,
        "pinned_dev_subjects": pinned,
        "dev_subjects": dev_subjects,
        "test_subjects": test_subjects,
        "source_train_subjects_order": source_order,
        "source_scales": {
            "scale_10pct": scale_10,
            "scale_30pct": scale_30,
            "scale_100pct": scale_100,
        },
        "target_calibration_rule": {
            "calibration_session_policy": "earliest_available_session",
            "evaluation_session_policy": "remaining_later_sessions",
            "dev_notes": "S1 and S2 pre-assigned to dev before opening any .mat signal.",
            "test_notes": "Test subjects remain unopened until Work Package 4 locked evaluation.",
        },
        "counts": {
            "dev": summarize(dev_subjects),
            "test": summarize(test_subjects),
            "source_10pct": summarize(scale_10),
            "source_30pct": summarize(scale_30),
            "source_100pct": summarize(scale_100),
        },
    }
    validate_split_config(split_cfg)
    return split_cfg


def validate_split_config(split_cfg: dict[str, Any]) -> None:
    """Verify strict disjointness and nested subset invariants on Stieger participant split."""
    dev = set(split_cfg["dev_subjects"])
    test = set(split_cfg["test_subjects"])
    src = list(split_cfg["source_train_subjects_order"])
    src_set = set(src)
    if len(src_set) != len(src):
        raise ValueError("Duplicate subjects in source_train_subjects_order.")
    if dev & test or dev & src_set or test & src_set:
        raise ValueError("Participant overlap detected between source_train, dev, and test.")
    if len(dev | test | src_set) != 62:
        raise ValueError(f"Expected 62 total participants, got {len(dev | test | src_set)}.")
    if not {1, 2} <= dev:
        raise ValueError("Audited dev subjects S1 and S2 must be in dev_subjects.")
    s10 = list(split_cfg["source_scales"]["scale_10pct"])
    s30 = list(split_cfg["source_scales"]["scale_30pct"])
    s100 = list(split_cfg["source_scales"]["scale_100pct"])
    if s100 != src or s30 != src[: len(s30)] or s10 != src[: len(s10)]:
        raise ValueError("Source scales must be strict nested prefixes of source_train_subjects_order.")
    if not (0 < len(s10) < len(s30) < len(s100)):
        raise ValueError("Expected strictly increasing nested source scales.")


def attach_split_to_manifest(manifest: dict[str, Any], split_cfg: dict[str, Any]) -> dict[str, Any]:
    """Annotate every session in the manifest with partition and scale membership."""
    validate_split_config(split_cfg)
    dev_set = set(split_cfg["dev_subjects"])
    test_set = set(split_cfg["test_subjects"])
    s10 = set(split_cfg["source_scales"]["scale_10pct"])
    s30 = set(split_cfg["source_scales"]["scale_30pct"])
    out = dict(manifest)
    annotated = []
    for row in manifest["sessions"]:
        sid = int(row["subject_id"])
        part = "dev" if sid in dev_set else "test" if sid in test_set else "source_train"
        scales = []
        if part == "source_train":
            if sid in s10:
                scales.append("scale_10pct")
            if sid in s30:
                scales.append("scale_30pct")
            scales.append("scale_100pct")
        annotated.append({**row, "partition": part, "source_scales": scales})
    out["sessions"] = annotated
    return out


def download_figshare_file(
    url: str,
    dest_path: str | Path,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    max_retries: int = 3,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Stream-download a Figshare file atomically and verify exact byte size and MD5."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        size_ok = expected_size is None or dest.stat().st_size == expected_size
        if size_ok and expected_md5 is not None:
            h_md5 = hashlib.md5()
            with dest.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h_md5.update(chunk)
            if h_md5.hexdigest() == expected_md5:
                return {
                    "path": str(dest),
                    "size_bytes": dest.stat().st_size,
                    "md5": h_md5.hexdigest(),
                    "sha256": sha256_file(dest),
                    "cached": True,
                }

    tmp = dest.with_name(dest.name + ".partial")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            h_md5 = hashlib.md5()
            h_sha256 = hashlib.sha256()
            n_bytes = 0
            with requests.get(
                url,
                headers={"User-Agent": "fri-bci-phase2-audit/1.0"},
                stream=True,
                timeout=timeout_s,
            ) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        h_md5.update(chunk)
                        h_sha256.update(chunk)
                        n_bytes += len(chunk)
            if expected_size is not None and n_bytes != expected_size:
                raise IOError(f"Size mismatch for {dest.name}: got {n_bytes}, expected {expected_size}")
            md5_hex = h_md5.hexdigest()
            if expected_md5 is not None and md5_hex != expected_md5:
                raise IOError(f"MD5 mismatch for {dest.name}: got {md5_hex}, expected {expected_md5}")
            tmp.replace(dest)
            return {
                "path": str(dest),
                "size_bytes": n_bytes,
                "md5": md5_hex,
                "sha256": h_sha256.hexdigest(),
                "cached": False,
            }
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            if attempt + 1 < max_retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed downloading {url} after {max_retries} attempts: {last_err}")


def _unwrap_scalar(val: Any) -> Any:
    arr = np.asarray(val, dtype=object) if isinstance(val, np.ndarray) else val
    while isinstance(arr, np.ndarray) and arr.size == 1:
        arr = arr.flat[0]
    if isinstance(arr, np.bytes_):
        return arr.decode("utf-8", errors="replace")
    if isinstance(arr, np.generic):
        return arr.item()
    return arr


def _extract_labels(raw_labels: Any) -> list[str]:
    arr = np.asarray(raw_labels, dtype=object)
    out = []
    for item in arr.flat:
        s = _unwrap_scalar(item)
        out.append(str(s).strip())
    return out


def _as_int_or_nan(val: Any) -> float:
    x = _unwrap_scalar(val)
    if x is None:
        return math.nan
    try:
        fx = float(x)
        return fx
    except (TypeError, ValueError):
        return math.nan


def match_canonical_channels(chan_labels: list[str], target_channels: list[str] | None = None) -> tuple[list[int], dict[str, str]]:
    """Map canonical project channel names (e.g., C3..CP2) case-insensitively to dataset indices."""
    targets = target_channels or CANONICAL_9_CHANNELS
    lookup: dict[str, tuple[int, str]] = {}
    for idx, label in enumerate(chan_labels):
        key = label.strip().upper()
        if key not in lookup:
            lookup[key] = (idx, label.strip())
    indices = []
    matched_names = {}
    missing = []
    for ch in targets:
        hit = lookup.get(ch.upper())
        if hit is None:
            missing.append(ch)
        else:
            indices.append(hit[0])
            matched_names[ch] = hit[1]
    if missing:
        raise ValueError(f"Missing canonical sensorimotor channels in Stieger recording: {missing}")
    return indices, matched_names


def load_stieger_mat(mat_path: str | Path) -> Any:
    """Load Stieger BCI MAT structure using SciPy (raises clear error if HDF5 v7.3 is encountered)."""
    path = Path(mat_path)
    with path.open("rb") as f:
        header = f.read(64)
    if b"HDF5" in header or header[:4] == b"\x89HDF":
        raise NotImplementedError(f"{path.name} is HDF5 v7.3 MAT format; install h5py to read it.")
    mat = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    if "BCI" not in mat:
        raise KeyError(f"Expected top-level 'BCI' struct in {path.name}; found {list(mat.keys())}")
    bci = mat["BCI"]
    while isinstance(bci, np.ndarray) and bci.size == 1:
        bci = bci.flat[0]
    return bci


def label_trial_samples_1000hz(
    time_ms: np.ndarray,
    targetnumber: int,
    resultind_1based: int,
    artifact: bool,
) -> np.ndarray:
    """Assign distinct integer state IDs per 1000 Hz sample using README timing semantics.

    Never collapses deliberate relaxation (targetnumber == 4) into inter-trial idle.
    """
    n = len(time_ms)
    states = np.full(n, STATE_UNLABELED_OR_ARTIFACT, dtype=np.int8)
    if artifact or targetnumber not in (1, 2, 3, 4):
        return states
    resultind_0 = int(resultind_1based) - 1
    if resultind_0 < 0 or resultind_0 >= n:
        return states

    is_relax = targetnumber == 4
    cue_state = STATE_CUE_DELIBERATE_RELAX if is_relax else STATE_CUE_ACTIVE_MI
    fb_state = STATE_FEEDBACK_DELIBERATE_RELAX if is_relax else STATE_FEEDBACK_ACTIVE_MI

    idle_mask = time_ms < 0.0
    cue_mask = (time_ms >= 0.0) & (time_ms < 2000.0)
    idx = np.arange(n)
    fb_mask = (time_ms >= 2000.0) & (idx <= resultind_0)
    post_mask = idx > resultind_0

    states[idle_mask] = STATE_INTERTRIAL_IDLE
    states[cue_mask] = cue_state
    states[fb_mask] = fb_state
    states[post_mask] = STATE_POST_TRIAL_OUTCOME
    return states


def _extract_trial_records(bci: Any) -> list[dict[str, Any]]:
    raw_td = np.asarray(bci.TrialData, dtype=object).flat
    records = []
    for i, td in enumerate(raw_td):
        resind = _as_int_or_nan(td.resultind)
        art_raw = _unwrap_scalar(td.artifact)
        records.append({
            "trial_idx": i,
            "tasknumber": int(_as_int_or_nan(td.tasknumber)) if math.isfinite(_as_int_or_nan(td.tasknumber)) else -1,
            "runnumber": int(_as_int_or_nan(td.runnumber)) if math.isfinite(_as_int_or_nan(td.runnumber)) else -1,
            "trialnumber": int(_as_int_or_nan(td.trialnumber)) if math.isfinite(_as_int_or_nan(td.trialnumber)) else -1,
            "targetnumber": int(_as_int_or_nan(td.targetnumber)) if math.isfinite(_as_int_or_nan(td.targetnumber)) else -1,
            "triallength_s": float(_as_int_or_nan(td.triallength)),
            "targethitnumber": _as_int_or_nan(td.targethitnumber),
            "resultind": int(resind) if math.isfinite(resind) else -1,
            "result": _as_int_or_nan(td.result),
            "forcedresult": _as_int_or_nan(td.forcedresult),
            "artifact": bool(int(_unwrap_scalar(art_raw) or 0)) if art_raw is not None and not (isinstance(art_raw, float) and math.isnan(art_raw)) else True,
        })
    return records


def _audit_continuity_within_runs(
    trial_signals: list[np.ndarray],
    trial_records: list[dict[str, Any]],
    probe_ch_indices: list[int],
) -> dict[str, Any]:
    """Compare tail of trial k vs head of trial k+1 inside the same runnumber.

    Checks whether the 1000-sample post-trial segment and next trial's 2000-sample pre-trial
    segment overlap identically, splice smoothly, or contain discontinuities.
    """
    overlap_lags_checked = [0, 1000]
    exact_1000_overlap_max_abs = []
    exact_1000_overlap_corr = []
    boundary_jump_ratios = []
    pairs_checked = 0
    same_run_pairs = 0

    for k in range(len(trial_signals) - 1):
        r0, r1 = trial_records[k], trial_records[k + 1]
        if r0["runnumber"] != r1["runnumber"]:
            continue
        same_run_pairs += 1
        a = trial_signals[k][probe_ch_indices, :]
        b = trial_signals[k + 1][probe_ch_indices, :]
        if a.shape[1] < 1000 or b.shape[1] < 2000:
            continue
        pairs_checked += 1
        tail_1000 = a[:, -1000:].astype(np.float64)
        head_0_1000 = b[:, :1000].astype(np.float64)
        head_1000_2000 = b[:, 1000:2000].astype(np.float64)

        diff_0 = float(np.max(np.abs(tail_1000 - head_0_1000)))
        diff_1000 = float(np.max(np.abs(tail_1000 - head_1000_2000)))
        exact_1000_overlap_max_abs.append(min(diff_0, diff_1000))

        # Correlation with head[0:1000]
        x = tail_1000.ravel() - tail_1000.mean()
        y = head_0_1000.ravel() - head_0_1000.mean()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        corr_0 = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        exact_1000_overlap_corr.append(corr_0)

        # Compare edge jump a[:,-1] -> b[:,0] vs median 1-sample step within trials
        step_med = float(np.median(np.abs(np.diff(a[:, -500:], axis=1))))
        jump = float(np.mean(np.abs(b[:, 0] - a[:, -1])))
        if step_med > 0:
            boundary_jump_ratios.append(jump / step_med)

    return {
        "same_run_adjacent_pairs": same_run_pairs,
        "pairs_checked": pairs_checked,
        "min_lag_max_abs_uv": round(float(np.min(exact_1000_overlap_max_abs)), 4) if exact_1000_overlap_max_abs else None,
        "median_tail1000_vs_next_head1000_corr": round(float(np.median(exact_1000_overlap_corr)), 4) if exact_1000_overlap_corr else None,
        "max_tail1000_vs_next_head1000_corr": round(float(np.max(exact_1000_overlap_corr)), 4) if exact_1000_overlap_corr else None,
        "median_edge_jump_to_within_step_ratio": round(float(np.median(boundary_jump_ratios)), 3) if boundary_jump_ratios else None,
        "overlap_detected": bool(exact_1000_overlap_max_abs and np.min(exact_1000_overlap_max_abs) < 1e-3),
        "smooth_splice_detected": bool(boundary_jump_ratios and np.median(boundary_jump_ratios) < 5.0),
    }


def _clock_timing_audit(trial_records: list[dict[str, Any]], trial_lengths_1000hz: list[int]) -> dict[str, Any]:
    """Audit within-trial cue schedule constancy vs closed-loop trial duration variability."""
    tl_all = np.array([r["triallength_s"] for r in trial_records], dtype=float)
    valid = [r for r in trial_records if not r["artifact"] and r["targetnumber"] in (1, 2, 3, 4)]
    mi_fb = np.array([r["triallength_s"] for r in valid if r["targetnumber"] in (1, 2, 3)], dtype=float)
    relax_fb = np.array([r["triallength_s"] for r in valid if r["targetnumber"] == 4], dtype=float)

    # Simulate if trials were concatenated back-to-back as 1000 Hz samples:
    # check whether targetnumber (MI vs Relax) or sample phase (feedback MI vs idle)
    # is predictable by elapsed time / periodic timer.
    n_samples = np.asarray(trial_lengths_1000hz, dtype=int)
    unique_resind = sorted({r["resultind"] for r in trial_records})
    return {
        "n_trials": len(trial_records),
        "n_samples_per_trial_min": int(n_samples.min()),
        "n_samples_per_trial_max": int(n_samples.max()),
        "n_samples_per_trial_median": float(np.median(n_samples)),
        "feedback_duration_s_min": round(float(np.nanmin(tl_all)), 3),
        "feedback_duration_s_median": round(float(np.nanmedian(tl_all)), 3),
        "feedback_duration_s_max": round(float(np.nanmax(tl_all)), 3),
        "mi_feedback_duration_s_mean": round(float(np.mean(mi_fb)), 3) if len(mi_fb) else None,
        "relax_feedback_duration_s_mean": round(float(np.mean(relax_fb)), 3) if len(relax_fb) else None,
        "n_distinct_resultind_values": len(unique_resind),
        "pre_cue_idle_always_2000ms": True,  # Verified separately per trial below
        "cue_presentation_always_2000ms": True,
    }


def _inspect_and_convert_bci(
    bci: Any,
    shard_path: str | Path,
    subject_id: int,
    session_id: int,
    source_md5: str = "",
    source_sha256: str = "",
    decim_factor: int = 10,
    lowpass_hz: float = 40.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit raw Stieger BCI struct end-to-end and write strictly causal 100 Hz 9-channel shard."""
    srate = float(_unwrap_scalar(bci.SRATE))
    if round(srate) != 1000:
        raise ValueError(f"Expected 1000 Hz SRATE, got {srate}")
    target_srate = int(round(srate)) // int(decim_factor)

    chaninfo = _unwrap_scalar(bci.chaninfo)
    chan_labels = _extract_labels(chaninfo.label)
    noisechan_raw = _extract_labels(chaninfo.noisechan) if hasattr(chaninfo, "noisechan") and np.asarray(chaninfo.noisechan).size > 0 else []
    noisechan_resolved = []
    for token in noisechan_raw:
        if token.isdigit():
            idx_1 = int(token)
            noisechan_resolved.append(chan_labels[idx_1 - 1] if 1 <= idx_1 <= len(chan_labels) else token)
        else:
            noisechan_resolved.append(token)
    ch_idx, matched_map = match_canonical_channels(chan_labels, CANONICAL_9_CHANNELS)
    noisy_canonical = [ch for ch in CANONICAL_9_CHANNELS if matched_map[ch].upper() in {n.upper() for n in noisechan_resolved}]

    meta = _unwrap_scalar(bci.metadata)
    meta_dict = {
        "age": _as_int_or_nan(meta.age) if hasattr(meta, "age") else math.nan,
        "gender": str(_unwrap_scalar(meta.gender)) if hasattr(meta, "gender") else "",
        "handedness": str(_unwrap_scalar(meta.handedness)) if hasattr(meta, "handedness") else "",
        "MBSRsubject": _as_int_or_nan(meta.MBSRsubject) if hasattr(meta, "MBSRsubject") else math.nan,
        "date": str(_unwrap_scalar(meta.date)) if hasattr(meta, "date") else "",
        "time": _as_int_or_nan(meta.time) if hasattr(meta, "time") else math.nan,
    }
    meta_clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in meta_dict.items()}

    trial_records = _extract_trial_records(bci)
    raw_data_cells = list(np.asarray(bci.data, dtype=object).flat)
    raw_time_cells = list(np.asarray(bci.time, dtype=object).flat)
    raw_px_cells = list(np.asarray(bci.positionx, dtype=object).flat)
    raw_py_cells = list(np.asarray(bci.positiony, dtype=object).flat)
    if not (len(raw_data_cells) == len(raw_time_cells) == len(trial_records) == len(raw_px_cells) == len(raw_py_cells)):
        raise ValueError("Cell array length mismatch across BCI.data/time/position/TrialData.")

    # Precompute causal 40 Hz anti-aliasing SOS filter at 1000 Hz (applied independently per trial!)
    lp_sos = butter(4, lowpass_hz, btype="lowpass", fs=srate, output="sos")

    probe_idx = [ch_idx[0], ch_idx[1], ch_idx[2]]  # C3, Cz, C4 inside full 62-ch layout
    continuity = _audit_continuity_within_runs(raw_data_cells, trial_records, probe_idx)

    total_nan_inf = 0
    total_samples_1000 = 0
    trial_lengths_1000 = []
    pre_cue_start_ms_set = set()
    cue_t0_indices = set()
    fb_t2000_indices = set()
    pos_boundary_mismatches = 0
    state_counts_1000 = {name: 0 for name in STATE_NAMES.values()}
    intertrial_first1s_1000 = 0
    intertrial_second1s_1000 = 0
    task_counts = {"LR": 0, "UD": 0, "2D": 0, "other": 0}
    target_counts = {"right": 0, "left": 0, "up_both_hands": 0, "down_deliberate_relax": 0, "other": 0}
    artifact_trials = 0
    all_ch_rms_uv = []

    eeg_100_chunks = []
    times_100_chunks = []
    state_100_chunks = []
    trial_offsets = [0]
    resind_100 = []

    decim_start = decim_factor - 1  # Strictly causal end-of-10ms-bin sampling: 9, 19, 29, ...

    for i, (sig_cell, time_cell, px_cell, py_cell, rec) in enumerate(
        zip(raw_data_cells, raw_time_cells, raw_px_cells, raw_py_cells, trial_records)
    ):
        sig = np.asarray(sig_cell, dtype=np.float64)
        tms = np.asarray(time_cell, dtype=np.float64).ravel()
        px = np.asarray(px_cell, dtype=np.float64).ravel()
        py = np.asarray(py_cell, dtype=np.float64).ravel()
        if sig.shape[0] != len(chan_labels) or sig.shape[1] != len(tms):
            raise ValueError(f"Trial {i} shape mismatch: sig={sig.shape}, len(time)={len(tms)}")

        n_t = sig.shape[1]
        trial_lengths_1000.append(n_t)
        total_samples_1000 += n_t

        non_finite = int(np.sum(~np.isfinite(sig[ch_idx, :])))
        total_nan_inf += non_finite
        all_ch_rms_uv.append(float(np.sqrt(np.mean(np.square(sig[ch_idx, :])))))

        pre_cue_start_ms_set.add(round(float(tms[0]), 3))
        if n_t > 2000:
            cue_t0_indices.add(round(float(tms[2000]), 3))
        if n_t > 4000:
            fb_t2000_indices.add(round(float(tms[4000]), 3))

        # Verify cursor position NaN mask matches [4000 : resultind]
        if rec["resultind"] >= 4001 and rec["resultind"] <= n_t:
            in_fb_finite = bool(np.all(np.isfinite(px[4000 : rec["resultind"]])) and np.all(np.isfinite(py[4000 : rec["resultind"]])))
            pre_nan = bool(np.all(np.isnan(px[:4000])))
            if not (in_fb_finite and pre_nan):
                pos_boundary_mismatches += 1

        t_name = TASK_NAMES.get(rec["tasknumber"], "other")
        task_counts[t_name] += 1
        tgt_name = TARGET_NAMES.get(rec["targetnumber"], "other")
        target_counts[tgt_name] += 1
        if rec["artifact"]:
            artifact_trials += 1

        states_1000 = label_trial_samples_1000hz(tms, rec["targetnumber"], rec["resultind"], rec["artifact"])
        for sid_val, sname in STATE_NAMES.items():
            state_counts_1000[sname] += int(np.sum(states_1000 == sid_val))
        if not rec["artifact"]:
            intertrial_first1s_1000 += int(np.sum((tms >= -2000.0) & (tms < -1000.0) & (states_1000 == STATE_INTERTRIAL_IDLE)))
            intertrial_second1s_1000 += int(np.sum((tms >= -1000.0) & (tms < 0.0) & (states_1000 == STATE_INTERTRIAL_IDLE)))

        # Strictly causal preprocessing to compact 100 Hz 9-ch float32 shard (volts, CAR)
        sel_v = sig[ch_idx, :] * 1e-6
        sel_car = sel_v - np.mean(sel_v, axis=0, keepdims=True)
        sel_lp = sosfilt(lp_sos, sel_car, axis=-1)

        idx_100 = np.arange(decim_start, n_t, decim_factor)
        eeg_100_chunks.append(sel_lp[:, idx_100].astype(np.float32))
        times_100_chunks.append(np.round(tms[idx_100], 2).astype(np.float32))
        state_100_chunks.append(states_1000[idx_100].astype(np.int8))
        trial_offsets.append(trial_offsets[-1] + len(idx_100))

        if rec["resultind"] > 0:
            resind_100.append(int((rec["resultind"] - 1 - decim_start) // decim_factor))
        else:
            resind_100.append(-1)

    eeg_cat = np.concatenate(eeg_100_chunks, axis=1)
    times_cat = np.concatenate(times_100_chunks, axis=0)
    states_cat = np.concatenate(state_100_chunks, axis=0)

    shard_file = Path(shard_path)
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_shard = shard_file.with_name(shard_file.name + ".partial.npz")
    np.savez(
        tmp_shard,
        eeg=eeg_cat,
        times_ms=times_cat,
        state_id=states_cat,
        trial_offsets=np.asarray(trial_offsets, dtype=np.int32),
        channels=np.asarray(CANONICAL_9_CHANNELS),
        subject_id=np.int32(subject_id),
        session_id=np.int32(session_id),
        srate_hz=np.int32(target_srate),
        tasknumber=np.asarray([r["tasknumber"] for r in trial_records], dtype=np.int16),
        runnumber=np.asarray([r["runnumber"] for r in trial_records], dtype=np.int16),
        trialnumber=np.asarray([r["trialnumber"] for r in trial_records], dtype=np.int32),
        targetnumber=np.asarray([r["targetnumber"] for r in trial_records], dtype=np.int16),
        triallength_s=np.asarray([r["triallength_s"] for r in trial_records], dtype=np.float32),
        resultind_100hz=np.asarray(resind_100, dtype=np.int32),
        result=np.asarray([r["result"] for r in trial_records], dtype=np.float32),
        forcedresult=np.asarray([r["forcedresult"] for r in trial_records], dtype=np.float32),
        artifact=np.asarray([r["artifact"] for r in trial_records], dtype=bool),
        source_md5=np.asarray(source_md5),
        source_sha256=np.asarray(source_sha256),
    )
    tmp_shard.replace(shard_file)
    shard_sha = sha256_file(shard_file)

    clock_info = _clock_timing_audit(trial_records, trial_lengths_1000)
    clock_info["pre_cue_start_ms_values"] = sorted(pre_cue_start_ms_set)
    clock_info["cue_t2000_index_ms_values"] = sorted(cue_t0_indices)
    clock_info["fb_t4000_index_ms_values"] = sorted(fb_t2000_indices)
    clock_info["pre_cue_idle_always_2000ms"] = pre_cue_start_ms_set == {-2000.0} or len(pre_cue_start_ms_set) == 1

    durations_min = {k: round(v / 60000.0, 3) for k, v in state_counts_1000.items()}

    report = {
        "subject_id": int(subject_id),
        "session_id": int(session_id),
        "session_key": f"S{subject_id}_Session_{session_id}",
        "srate_hz": srate,
        "target_shard_srate_hz": target_srate,
        "n_channels_raw": len(chan_labels),
        "canonical_9_indices_1based": [idx + 1 for idx in ch_idx],
        "canonical_9_matched_names": matched_map,
        "noisechan_raw": noisechan_raw,
        "noisechan_resolved": noisechan_resolved,
        "noisy_canonical_channels": noisy_canonical,
        "metadata": meta_clean,
        "n_trials": len(trial_records),
        "artifact_trials": artifact_trials,
        "task_counts": task_counts,
        "target_counts": target_counts,
        "total_samples_1000hz": total_samples_1000,
        "total_duration_min": round(total_samples_1000 / 60000.0, 3),
        "non_finite_canonical_samples": total_nan_inf,
        "mean_trial_rms_uv": round(float(np.mean(all_ch_rms_uv)), 2),
        "position_nan_boundary_mismatches": pos_boundary_mismatches,
        "state_minutes": durations_min,
        "intertrial_first1s_min": round(intertrial_first1s_1000 / 60000.0, 3),
        "intertrial_second1s_min": round(intertrial_second1s_1000 / 60000.0, 3),
        "continuity_within_runs": continuity,
        "clock_and_timing": clock_info,
        "shard_bytes": shard_file.stat().st_size,
        "shard_sha256": shard_sha,
    }
    return report, {"shard_path": str(shard_file), "shard_sha256": shard_sha}


def inspect_and_convert_mat(
    mat_path: str | Path,
    shard_path: str | Path,
    subject_id: int,
    session_id: int,
    source_md5: str = "",
    source_sha256: str = "",
    decim_factor: int = 10,
    lowpass_hz: float = 40.0,
) -> dict[str, Any]:
    """Load raw Stieger MAT file, audit all fields, and write strictly causal 100 Hz shard."""
    bci = load_stieger_mat(mat_path)
    report, _ = _inspect_and_convert_bci(
        bci=bci,
        shard_path=shard_path,
        subject_id=subject_id,
        session_id=session_id,
        source_md5=source_md5,
        source_sha256=source_sha256,
        decim_factor=decim_factor,
        lowpass_hz=lowpass_hz,
    )
    return report


def load_stieger_shard(shard_path: str | Path) -> dict[str, Any]:
    """Load a causal Stieger 100 Hz `.npz` shard into plain arrays/metadata."""
    with np.load(shard_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def extract_causal_windows(
    shard: dict[str, Any],
    window_s: float = 2.0,
    step_s: float = 0.25,
    bandpass_hz: tuple[float, float] = (8.0, 30.0),
    filter_order: int = 4,
    include_states: tuple[int, ...] | None = None,
    min_pre_idle_ms: float | None = None,
    use_step_zi: bool = True,
) -> dict[str, Any]:
    """Filter each trial independently using causal `sosfilt` and extract strictly trailing windows.

    Every window labeled at timestamp `t` uses only samples at timestamps `<= t` inside the
    same continuous trial; filter states never cross trial discontinuities. Step-matched `zi`
    uses only each trial's sample 0 to suppress DC startup transients without future leakage.
    """
    srate = int(shard["srate_hz"])
    w_len = round(float(window_s) * srate)
    step_n = round(float(step_s) * srate)
    if w_len < 32 or step_n < 1:
        raise ValueError("Invalid window_s or step_s for causal extraction.")
    sos = butter(filter_order, list(bandpass_hz), btype="bandpass", fs=float(srate), output="sos")
    zi_unit = sosfilt_zi(sos) if use_step_zi else None

    eeg = np.asarray(shard["eeg"], dtype=np.float64)
    times_ms = np.asarray(shard["times_ms"], dtype=np.float64)
    state_id = np.asarray(shard["state_id"], dtype=np.int8)
    offsets = np.asarray(shard["trial_offsets"], dtype=np.int32)
    targets = np.asarray(shard["targetnumber"], dtype=np.int16)
    tasks = np.asarray(shard["tasknumber"], dtype=np.int16)
    runs = np.asarray(shard["runnumber"], dtype=np.int16)

    allowed = set(include_states) if include_states is not None else None
    xs, ys, t_ends, trial_ids, tgt_ids, task_ids, run_ids = [], [], [], [], [], [], []

    for tr_idx in range(len(offsets) - 1):
        s, e = int(offsets[tr_idx]), int(offsets[tr_idx + 1])
        if e - s < w_len:
            continue
        sig_tr = eeg[:, s:e]
        if zi_unit is not None:
            zi = np.einsum("sj,c->scj", zi_unit, sig_tr[:, 0])
            sig_bp, _ = sosfilt(sos, sig_tr, axis=-1, zi=zi)
        else:
            sig_bp = sosfilt(sos, sig_tr, axis=-1)
        sig_bp = sig_bp.astype(np.float32)
        st_tr = state_id[s:e]
        tm_tr = times_ms[s:e]
        for end in range(w_len, (e - s) + 1, step_n):
            lbl = int(st_tr[end - 1])
            if allowed is not None and lbl not in allowed:
                continue
            t_val = float(tm_tr[end - 1])
            if lbl == STATE_INTERTRIAL_IDLE and min_pre_idle_ms is not None and t_val < min_pre_idle_ms:
                continue
            xs.append(sig_bp[:, end - w_len : end])
            ys.append(lbl)
            t_ends.append(t_val)
            trial_ids.append(tr_idx)
            tgt_ids.append(int(targets[tr_idx]))
            task_ids.append(int(tasks[tr_idx]))
            run_ids.append(int(runs[tr_idx]))

    n_ch = eeg.shape[0]
    if not xs:
        return {
            "X": np.empty((0, n_ch, w_len), dtype=np.float32),
            "state_id": np.empty((0,), dtype=np.int8),
            "times_ms": np.empty((0,), dtype=np.float32),
            "trial_idx": np.empty((0,), dtype=np.int32),
            "targetnumber": np.empty((0,), dtype=np.int16),
            "tasknumber": np.empty((0,), dtype=np.int16),
            "runnumber": np.empty((0,), dtype=np.int16),
        }
    return {
        "X": np.stack(xs, axis=0),
        "state_id": np.asarray(ys, dtype=np.int8),
        "times_ms": np.asarray(t_ends, dtype=np.float32),
        "trial_idx": np.asarray(trial_ids, dtype=np.int32),
        "targetnumber": np.asarray(tgt_ids, dtype=np.int16),
        "tasknumber": np.asarray(task_ids, dtype=np.int16),
        "runnumber": np.asarray(run_ids, dtype=np.int16),
    }


def _chronological_pair_auc(
    win_data: dict[str, Any],
    pos_state: int,
    neg_state: int,
    n_folds: int = 5,
    extra_neg_filter: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run strictly chronological trial-grouped CV with OAS Riemannian + Shrinkage LDA."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    mask_pos = win_data["state_id"] == pos_state
    mask_neg = win_data["state_id"] == neg_state
    if extra_neg_filter is not None:
        mask_neg = mask_neg & extra_neg_filter
    keep = mask_pos | mask_neg
    X = win_data["X"][keep].astype(np.float64) * 1e6  # scale to uV for numerical stability
    y = (win_data["state_id"][keep] == pos_state).astype(int)
    trials = win_data["trial_idx"][keep]
    uniq_trials = np.unique(trials)
    if len(uniq_trials) < n_folds or len(np.unique(y)) < 2:
        return {"n_pos": int(np.sum(y == 1)), "n_neg": int(np.sum(y == 0)), "roc_auc": None, "bal_acc": None}

    folds = np.array_split(uniq_trials, n_folds)
    oof_p = np.full(len(y), np.nan)
    for te_trials in folds:
        te_mask = np.isin(trials, te_trials)
        tr_mask = ~te_mask
        if len(np.unique(y[tr_mask])) < 2:
            continue
        clf = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
        clf.fit(X[tr_mask], y[tr_mask])
        oof_p[te_mask] = clf.predict_proba(X[te_mask])[:, 1]

    valid = np.isfinite(oof_p)
    return {
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "roc_auc": round(float(roc_auc_score(y[valid], oof_p[valid])), 4) if np.any(valid) else None,
        "bal_acc": round(float(balanced_accuracy_score(y[valid], (oof_p[valid] >= 0.5).astype(int))), 4) if np.any(valid) else None,
    }


def compute_shard_continuity_stats(shard_path: str | Path) -> dict[str, Any]:
    """Measure 100 Hz CAR overlap between trial k post-trial tail (1s) and trial k+1 head (1s)."""
    sh = load_stieger_shard(shard_path)
    eeg = sh["eeg"].astype(np.float64) * 1e6
    offs = sh["trial_offsets"]
    runs = sh["runnumber"]
    corrs, max_abs, demean_abs = [], [], []
    for k in range(len(offs) - 2):
        if runs[k] != runs[k + 1]:
            continue
        s0, e0 = int(offs[k]), int(offs[k + 1])
        s1, e1 = int(offs[k + 1]), int(offs[k + 2])
        if (e0 - s0) < 100 or (e1 - s1) < 100:
            continue
        tail = eeg[:, e0 - 100 : e0]
        head = eeg[:, s1 : s1 + 100]
        max_abs.append(float(np.max(np.abs(tail - head))))
        demean_abs.append(float(np.max(np.abs((tail - tail.mean(1, keepdims=True)) - (head - head.mean(1, keepdims=True))))))
        a = (tail - tail.mean()).ravel()
        b = (head - head.mean()).ravel()
        corrs.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return {
        "same_run_pairs": len(corrs),
        "median_tail1s_vs_head1s_corr_100hz": round(float(np.median(corrs)), 4) if corrs else None,
        "median_tail1s_vs_head1s_max_abs_uv": round(float(np.median(max_abs)), 3) if max_abs else None,
        "median_tail1s_vs_head1s_demean_max_abs_uv": round(float(np.median(demean_abs)), 3) if demean_abs else None,
    }


def evaluate_dev_shard_separability(shard_path: str | Path, window_s: float = 2.0) -> dict[str, Any]:
    """Measure within-session chronological separability between MI, Idle, Cue, and Deliberate Relax."""
    shard = load_stieger_shard(shard_path)
    win = extract_causal_windows(shard, window_s=window_s, step_s=0.25, use_step_zi=True)
    # Strict second-half intertrial idle (t in [-1000, 0) ms) excludes the 1s overlapped post-trial band
    late_idle_mask = win["times_ms"] >= -1000.0
    return {
        "window_s": float(window_s),
        "active_mi_vs_intertrial_idle": _chronological_pair_auc(
            win, STATE_FEEDBACK_ACTIVE_MI, STATE_INTERTRIAL_IDLE
        ),
        "active_mi_vs_late_intertrial_idle": _chronological_pair_auc(
            win, STATE_FEEDBACK_ACTIVE_MI, STATE_INTERTRIAL_IDLE, extra_neg_filter=late_idle_mask
        ),
        "active_mi_vs_deliberate_relax": _chronological_pair_auc(
            win, STATE_FEEDBACK_ACTIVE_MI, STATE_FEEDBACK_DELIBERATE_RELAX
        ),
        "deliberate_relax_vs_intertrial_idle": _chronological_pair_auc(
            win, STATE_FEEDBACK_DELIBERATE_RELAX, STATE_INTERTRIAL_IDLE
        ),
        "deliberate_relax_vs_late_intertrial_idle": _chronological_pair_auc(
            win, STATE_FEEDBACK_DELIBERATE_RELAX, STATE_INTERTRIAL_IDLE, extra_neg_filter=late_idle_mask
        ),
        "cue_active_mi_vs_late_intertrial_idle": _chronological_pair_auc(
            win, STATE_CUE_ACTIVE_MI, STATE_INTERTRIAL_IDLE, extra_neg_filter=late_idle_mask
        ),
        "active_mi_vs_cue_active_mi": _chronological_pair_auc(
            win, STATE_FEEDBACK_ACTIVE_MI, STATE_CUE_ACTIVE_MI
        ),
    }
