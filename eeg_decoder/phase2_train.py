"""Resumable Phase 2 EEG pretraining, Drive checkpoint verification, and Figshare shard CLI.

Supports:
  - `ingest`: download/verify Figshare sessions, convert to causal 100 Hz 9-ch shards, mirror to Drive.
  - `train`: 3-class causal EEGNet source pretraining (clean_idle vs deliberate_relax vs active_mi)
             with GPU/AMP auto-profiling, atomic checkpoints, SHA-256 Drive verification, SIGINT resume.
  - `status` / `next-job`: inspect/advance the nested 10% / 30% / 100% x 3-seed Colab campaign ledger.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

from .alignment import apply_alignment, fit_alignment
from .neural import EEGNet
from .stieger import (
    STATE_FEEDBACK_ACTIVE_MI,
    STATE_FEEDBACK_DELIBERATE_RELAX,
    STATE_INTERTRIAL_IDLE,
    download_figshare_file,
    extract_causal_windows,
    inspect_and_convert_mat,
    load_stieger_shard,
    sha256_file,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CFG = REPO_ROOT / "configs" / "phase2_stieger_split.json"
DEFAULT_TRAIN_CFG = REPO_ROOT / "configs" / "phase2_training.json"
DEFAULT_MANIFEST = REPO_ROOT / "results" / "phase2_stieger_audit_2026-11-19" / "stieger_figshare_manifest.json"


class Phase2EEGNet(nn.Module):
    """Causal EEGNet8 wrapper supporting 3-class Stieger pretraining without altering neural.py."""

    def __init__(
        self,
        channels: int = 9,
        samples: int = 200,
        f1: int = 8,
        depth: int = 2,
        f2: int = 16,
        dropout: float = 0.25,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = EEGNet(channels=channels, samples=samples, f1=f1, depth=depth, f2=f2, dropout=dropout)
        if n_classes != 2:
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def constrain_weights(self) -> None:
        self.backbone.constrain_weights()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_compute_profile(requested_device: str = "auto", override_batch_size: int | None = None, config: dict | None = None) -> dict[str, Any]:
    """Inspect GPU/CPU runtime and pick safe batch size and mixed-precision settings."""
    cuda_ok = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA requested but unavailable; select a GPU runtime or pass --device cpu.")
    device = torch.device("cuda" if (requested_device == "auto" and cuda_ok) or requested_device == "cuda" else "cpu")

    bs_map = (config or {}).get("source_pretraining", {}).get("default_batch_size_by_gpu", {})
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gib = round(props.total_memory / (1024**3), 2)
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        amp_mode = "bf16" if bf16_ok else "fp16"
        default_bs = bs_map.get("T4", 128)
        for tag in ("H100", "A100", "L4", "T4"):
            if tag in gpu_name.upper():
                default_bs = int(bs_map.get(tag, default_bs))
                break
    else:
        gpu_name = "CPU"
        vram_gib = 0.0
        amp_mode = "fp32"
        default_bs = int(bs_map.get("CPU", 64))

    batch_size = int(override_batch_size) if override_batch_size is not None else int(default_bs)
    return {
        "device": device,
        "device_name": gpu_name,
        "vram_gib": vram_gib,
        "amp_mode": amp_mode,
        "batch_size": batch_size,
    }


def copy_with_sha256_verify(src: str | Path, dst: str | Path) -> str:
    """Atomically copy file to destination (e.g., Google Drive) and verify exact SHA-256."""
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(src_p)
    tmp = dst_p.with_name(dst_p.name + ".partial")
    shutil.copy2(src_p, tmp)
    tmp.replace(dst_p)
    actual = sha256_file(dst_p)
    if actual != expected:
        dst_p.unlink(missing_ok=True)
        raise IOError(f"Drive copy verification failed for {dst_p}: expected {expected}, got {actual}")
    return actual


def _pair_auc_safe(y_true: np.ndarray, probs: np.ndarray, pos_cls: int, neg_cls: int) -> float | None:
    mask = (y_true == pos_cls) | (y_true == neg_cls)
    if not np.any(mask):
        return None
    ym = (y_true[mask] == pos_cls).astype(int)
    if len(np.unique(ym)) < 2:
        return None
    denom = probs[mask, pos_cls] + probs[mask, neg_cls] + 1e-12
    score = probs[mask, pos_cls] / denom
    return round(float(roc_auc_score(ym, score)), 4)


def prepare_session_tensors(
    shard_path: str | Path,
    window_s: float = 2.0,
    step_s: float = 0.25,
    bandpass_hz: tuple[float, float] = (8.0, 30.0),
    clean_idle_min_ms: float = -1000.0,
    alignment_matrix: np.ndarray | None = None,
    causal_run1_align: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract causal 3-class windows (0=clean_idle, 1=deliberate_relax, 2=active_mi) and apply EA.

    Never fits EA across future evaluation runs: if `causal_run1_align=True` and no earlier
    calibration `alignment_matrix` was supplied, fits EA strictly on `runnumber == min_run`
    and returns only windows from `runnumber > min_run`.
    """
    sh = load_stieger_shard(shard_path)
    win = extract_causal_windows(
        sh,
        window_s=window_s,
        step_s=step_s,
        bandpass_hz=bandpass_hz,
        include_states=(STATE_INTERTRIAL_IDLE, STATE_FEEDBACK_DELIBERATE_RELAX, STATE_FEEDBACK_ACTIVE_MI),
        min_pre_idle_ms=clean_idle_min_ms,
        use_step_zi=True,
    )
    if len(win["X"]) == 0:
        w_len = round(float(window_s) * int(sh["srate_hz"]))
        return np.empty((0, sh["eeg"].shape[0], w_len), dtype=np.float32), np.empty((0,), dtype=np.int64), alignment_matrix

    label_map = {
        STATE_INTERTRIAL_IDLE: 0,
        STATE_FEEDBACK_DELIBERATE_RELAX: 1,
        STATE_FEEDBACK_ACTIVE_MI: 2,
    }
    y = np.vectorize(label_map.__getitem__, otypes=[np.int64])(win["state_id"])
    X = win["X"]
    runs = win["runnumber"]

    if alignment_matrix is not None:
        return apply_alignment(X, alignment_matrix), y, alignment_matrix

    if causal_run1_align and len(np.unique(runs)) >= 2:
        min_run = int(np.min(runs))
        cal_mask = runs == min_run
        eval_mask = runs > min_run
        mat = fit_alignment(X[cal_mask])
        return apply_alignment(X[eval_mask], mat), y[eval_mask], mat

    mat = fit_alignment(X)
    return apply_alignment(X, mat), y, mat


def load_dataset_pools(
    train_shard_paths: list[Path],
    val_shard_paths: list[Path],
    window_s: float = 2.0,
    step_s: float = 0.25,
    bandpass_hz: tuple[float, float] = (8.0, 30.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build training and validation tensors with strict session/subject alignment hygiene."""
    tr_x, tr_y = [], []
    by_subj_cal: dict[int, np.ndarray] = {}

    train_sorted = sorted(train_shard_paths, key=lambda p: p.name)
    for p in train_sorted:
        sh = load_stieger_shard(p)
        sid = int(sh["subject_id"])
        sess_id = int(sh["session_id"])
        X_s, y_s, mat = prepare_session_tensors(p, window_s=window_s, step_s=step_s, bandpass_hz=bandpass_hz)
        if len(y_s) > 0:
            tr_x.append(X_s)
            tr_y.append(y_s)
        if sid not in by_subj_cal and mat is not None:
            by_subj_cal[sid] = mat

    va_x, va_y = [], []
    # Group validation shards by subject; earliest session of a val subject serves as calibration if >=2 exist
    val_by_subj: dict[int, list[tuple[int, Path]]] = {}
    for p in val_shard_paths:
        sh = load_stieger_shard(p)
        val_by_subj.setdefault(int(sh["subject_id"]), []).append((int(sh["session_id"]), p))

    for sid, items in sorted(val_by_subj.items()):
        items.sort(key=lambda t: t[0])
        if len(items) >= 2:
            # Earliest validation subject session calibrates alignment strictly; remaining sessions are evaluated
            _, cal_p = items[0]
            _, _, cal_mat = prepare_session_tensors(cal_p, window_s=window_s, step_s=step_s, bandpass_hz=bandpass_hz)
            for _, eval_p in items[1:]:
                X_v, y_v, _ = prepare_session_tensors(
                    eval_p, window_s=window_s, step_s=step_s, bandpass_hz=bandpass_hz, alignment_matrix=cal_mat
                )
                if len(y_v) > 0:
                    va_x.append(X_v)
                    va_y.append(y_v)
        else:
            # Single session in smoke/synthetic run: strictly fit EA on Run 1 only, evaluate Runs >= 2
            _, single_p = items[0]
            X_v, y_v, _ = prepare_session_tensors(
                single_p, window_s=window_s, step_s=step_s, bandpass_hz=bandpass_hz, causal_run1_align=True
            )
            if len(y_v) > 0:
                va_x.append(X_v)
                va_y.append(y_v)

    X_train = np.concatenate(tr_x, axis=0)
    y_train = np.concatenate(tr_y, axis=0)
    X_val = np.concatenate(va_x, axis=0)
    y_val = np.concatenate(va_y, axis=0)
    meta = {
        "n_train_shards": len(train_shard_paths),
        "n_val_shards": len(val_shard_paths),
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "train_class_counts": np.bincount(y_train, minlength=3).tolist(),
        "val_class_counts": np.bincount(y_val, minlength=3).tolist(),
        "window_samples": int(X_train.shape[-1]),
        "channels": int(X_train.shape[1]),
    }
    return X_train, y_train, X_val, y_val, meta


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, class_weights: torch.Tensor, device: torch.device, batch_size: int = 512) -> dict[str, Any]:
    """Compute validation CE loss, 3-class balanced accuracy, and clean diagnostic pair AUCs."""
    model.eval()
    probs_chunks = []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(device)
            yb = torch.from_numpy(np.asarray(y[i : i + batch_size], dtype=np.int64)).to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            probs_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs_chunks, axis=0)
    preds = np.argmax(probs, axis=1)
    bal_acc = round(float(balanced_accuracy_score(y, preds)), 4)
    auc_mi_idle = _pair_auc_safe(y, probs, 2, 0)
    auc_mi_relax = _pair_auc_safe(y, probs, 2, 1)
    auc_relax_idle = _pair_auc_safe(y, probs, 1, 0)
    valid_aucs = [v for v in (auc_mi_idle, auc_mi_relax) if v is not None]
    auc_macro = round(float(sum(valid_aucs) / len(valid_aucs)), 4) if valid_aucs else None
    return {
        "val_loss": round(total_loss / max(1, len(y)), 5),
        "val_bal_acc_3class": bal_acc,
        "val_auc_mi_vs_clean_idle": auc_mi_idle,
        "val_auc_mi_vs_deliberate_relax": auc_mi_relax,
        "val_auc_relax_vs_clean_idle": auc_relax_idle,
        "val_auc_macro_mi": auc_macro,
    }


def _rng_state_dict() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def _atomic_save_ckpt(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(payload, tmp)
    tmp.replace(path)
    return sha256_file(path)


def _sync_ledger(work_root: Path, drive_root: Path | None, job_id: str, job_state: dict[str, Any]) -> None:
    for root in [p for p in (work_root, drive_root) if p is not None]:
        ledger_path = root / "campaign_ledger.json"
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        else:
            ledger = {"schema_version": "phase2-campaign-ledger-v1", "jobs": {}}
        ledger.setdefault("jobs", {})[job_id] = job_state
        write_json(ledger_path, ledger)


def build_campaign_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    pre = config["source_pretraining"]
    win_s = float(config["preprocessing"]["primary_window_s"])
    max_updates = int(pre["matched_update_budget"])
    jobs = []
    for scale in pre["scales"]:
        for seed in pre["seeds"]:
            jobs.append({
                "job_id": f"{scale}_seed{seed}_w{win_s:.1f}s",
                "scale": scale,
                "seed": int(seed),
                "window_s": win_s,
                "max_updates": max_updates,
            })
    return jobs


def cmd_ingest(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    split_cfg = json.loads(Path(args.split_config).read_text(encoding="utf-8"))
    shard_dir = Path(args.shard_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else (shard_dir.parent / "raw")
    drive_shard_dir = Path(args.drive_shard_dir) if args.drive_shard_dir else None
    shard_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_rows = []
    if args.sessions:
        want = set(args.sessions)
        target_rows = [r for r in manifest["sessions"] if r["session_key"] in want]
        if len(target_rows) != len(want):
            missing = want - {r["session_key"] for r in target_rows}
            raise KeyError(f"Sessions not found in manifest: {sorted(missing)}")
    else:
        want_subs = set()
        if args.scale:
            want_subs |= set(split_cfg["source_scales"][args.scale])
        if args.include_dev:
            want_subs |= set(split_cfg["dev_subjects"])
        if not want_subs:
            raise ValueError("Specify --sessions or --scale and/or --include-dev.")
        target_rows = [r for r in manifest["sessions"] if int(r["subject_id"]) in want_subs]

    if args.max_sessions is not None:
        target_rows = target_rows[: args.max_sessions]

    shard_manifest_path = shard_dir / "shard_manifest.json"
    if shard_manifest_path.is_file():
        s_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    else:
        s_manifest = {"schema_version": "phase2-stieger-shards-v1", "shards": {}}

    drive_manifest = {}
    if drive_shard_dir and (drive_shard_dir / "shard_manifest.json").is_file():
        drive_manifest = json.loads((drive_shard_dir / "shard_manifest.json").read_text(encoding="utf-8")).get("shards", {})

    print(f"Staging {len(target_rows)} sessions into {shard_dir}...", flush=True)
    for idx, meta in enumerate(target_rows, start=1):
        key = meta["session_key"]
        local_shard = shard_dir / f"{key}.npz"
        expected_drive = drive_manifest.get(key, {}).get("shard_sha256")
        if local_shard.is_file():
            cur_sha = sha256_file(local_shard)
            rec_sha = s_manifest["shards"].get(key, {}).get("shard_sha256")
            if rec_sha in (None, cur_sha):
                s_manifest["shards"].setdefault(key, {
                    "session_key": key,
                    "subject_id": meta["subject_id"],
                    "session_id": meta["session_id"],
                    "partition": meta["partition"],
                    "shard_bytes": local_shard.stat().st_size,
                    "shard_sha256": cur_sha,
                    "source_md5": meta["computed_md5"],
                })
                continue

        if drive_shard_dir and (drive_shard_dir / f"{key}.npz").is_file():
            drive_src = drive_shard_dir / f"{key}.npz"
            copied_sha = copy_with_sha256_verify(drive_src, local_shard)
            if expected_drive and copied_sha != expected_drive:
                raise IOError(f"Drive shard SHA mismatch for {key}: expected {expected_drive}, got {copied_sha}")
            s_manifest["shards"][key] = {
                "session_key": key,
                "subject_id": meta["subject_id"],
                "session_id": meta["session_id"],
                "partition": meta["partition"],
                "shard_bytes": local_shard.stat().st_size,
                "shard_sha256": copied_sha,
                "source_md5": meta["computed_md5"],
            }
            print(f"[{idx}/{len(target_rows)}] Restored verified shard {key} from Drive", flush=True)
            continue

        free_gib = shutil.disk_usage(raw_dir).free / (1024**3)
        if free_gib < 1.2:
            raise RuntimeError(f"Low disk space ({free_gib:.2f} GiB free) in {raw_dir}; aborting safely.")
        mat_path = raw_dir / meta["filename"]
        dl = download_figshare_file(
            meta["download_url"], mat_path, expected_size=meta["size_bytes"], expected_md5=meta["computed_md5"]
        )
        rep = inspect_and_convert_mat(
            mat_path, local_shard, meta["subject_id"], meta["session_id"], dl["md5"], dl["sha256"]
        )
        gc.collect()
        if not args.keep_mat and mat_path.is_file():
            mat_path.unlink()
        if drive_shard_dir:
            copy_with_sha256_verify(local_shard, drive_shard_dir / f"{key}.npz")
        s_manifest["shards"][key] = {
            "session_key": key,
            "subject_id": meta["subject_id"],
            "session_id": meta["session_id"],
            "partition": meta["partition"],
            "shard_bytes": rep["shard_bytes"],
            "shard_sha256": rep["shard_sha256"],
            "source_md5": dl["md5"],
            "source_sha256": dl["sha256"],
        }
        write_json(shard_manifest_path, s_manifest)
        if drive_shard_dir:
            copy_with_sha256_verify(shard_manifest_path, drive_shard_dir / "shard_manifest.json")
        print(f"[{idx}/{len(target_rows)}] Converted & verified {key} ({rep['shard_bytes']/1e6:.2f} MB)", flush=True)

    write_json(shard_manifest_path, s_manifest)
    if drive_shard_dir:
        copy_with_sha256_verify(shard_manifest_path, drive_shard_dir / "shard_manifest.json")
    print(f"Staging complete: {len(target_rows)} sessions ready in {shard_dir}", flush=True)
    return 0


def _resolve_job_shards(args: argparse.Namespace, split_cfg: dict[str, Any], manifest: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    shard_dir = Path(args.shard_dir)
    if args.train_shards and args.val_shards:
        return [Path(p) for p in args.train_shards], [Path(p) for p in args.val_shards]
    if not args.scale:
        raise ValueError("Provide either --scale or explicit --train-shards and --val-shards.")
    src_subs = set(split_cfg["source_scales"][args.scale])
    dev_subs = set(split_cfg["dev_subjects"])
    train_paths = [shard_dir / f"{r['session_key']}.npz" for r in manifest["sessions"] if int(r["subject_id"]) in src_subs]
    val_paths = [shard_dir / f"{r['session_key']}.npz" for r in manifest["sessions"] if int(r["subject_id"]) in dev_subs]
    train_existing = [p for p in train_paths if p.is_file()]
    val_existing = [p for p in val_paths if p.is_file()]
    if len(train_existing) != len(train_paths):
        missing = [p.name for p in train_paths if not p.is_file()]
        raise FileNotFoundError(f"Missing {len(missing)} source shards for {args.scale} in {shard_dir} (first: {missing[:3]})")
    if len(val_existing) < 2:
        raise FileNotFoundError(f"Need at least 2 dev shards in {shard_dir}; run `ingest --include-dev` first.")
    return train_existing, val_existing


def cmd_train(args: argparse.Namespace) -> int:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    split_cfg = json.loads(Path(args.split_config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    pre = cfg["source_pretraining"]
    prep = cfg["preprocessing"]
    window_s = float(args.window_s if args.window_s is not None else prep["primary_window_s"])
    max_updates = int(args.max_updates if args.max_updates is not None else pre["matched_update_budget"])
    ckpt_every = int(args.checkpoint_every if args.checkpoint_every is not None else pre["checkpoint_every_updates"])
    eval_every = int(args.eval_every if args.eval_every is not None else pre["eval_every_updates"])
    lr = float(args.lr if args.lr is not None else pre["lr"])
    seed = int(args.seed)
    job_id = args.job_id or f"{args.scale or 'custom'}_seed{seed}_w{window_s:.1f}s"

    train_shards, val_shards = _resolve_job_shards(args, split_cfg, manifest)
    profile = detect_compute_profile(args.device, args.batch_size, cfg)
    device = profile["device"]
    batch_size = profile["batch_size"]
    amp_mode = profile["amp_mode"]

    print(
        f"Job={job_id} | Device={profile['device_name']} ({profile['vram_gib']} GiB, amp={amp_mode}, bs={batch_size}) "
        f"| Train shards={len(train_shards)}, Val shards={len(val_shards)}",
        flush=True,
    )

    t0_data = time.perf_counter()
    X_train, y_train, X_val, y_val, data_meta = load_dataset_pools(
        train_shards, val_shards, window_s=window_s, step_s=float(prep["step_s"]), bandpass_hz=tuple(prep["bandpass_hz"])
    )
    data_load_s = round(time.perf_counter() - t0_data, 3)

    counts = np.maximum(1, np.bincount(y_train, minlength=3)).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(counts)
    class_weights = torch.tensor(inv_sqrt / inv_sqrt.mean(), dtype=torch.float32)

    arch = pre["architecture"]
    seed_all(seed)
    model = Phase2EEGNet(
        channels=data_meta["channels"],
        samples=data_meta["window_samples"],
        f1=int(arch["f1"]),
        depth=int(arch["depth"]),
        f2=int(arch["f2"]),
        dropout=float(arch["dropout"]),
        n_classes=int(arch["n_classes"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(pre["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_updates))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_mode == "fp16"))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    work_root = Path(args.work_dir)
    job_dir = work_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    drive_root = Path(args.drive_dir) if args.drive_dir else None
    drive_job_dir = (drive_root / job_id) if drive_root else None
    ckpt_latest = job_dir / "ckpt_latest.pt"
    ckpt_best = job_dir / "ckpt_best.pt"
    state_path = job_dir / "job_state.json"

    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    manifest_hash = sha256_file(args.manifest)

    step, epoch, batch_offset, examples_seen = 0, 0, 0, 0
    best_score = -1.0
    history: list[dict[str, Any]] = []
    last_val_metrics: dict[str, Any] = {}

    if args.resume:
        if not ckpt_latest.is_file() and drive_job_dir and (drive_job_dir / "ckpt_latest.pt").is_file():
            copy_with_sha256_verify(drive_job_dir / "ckpt_latest.pt", ckpt_latest)
            if (drive_job_dir / "ckpt_best.pt").is_file():
                copy_with_sha256_verify(drive_job_dir / "ckpt_best.pt", ckpt_best)
        if ckpt_latest.is_file():
            ckpt = torch.load(ckpt_latest, map_location=device, weights_only=False)
            if int(ckpt["window_samples"]) != data_meta["window_samples"]:
                raise ValueError("Checkpoint window length mismatch on resume.")
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if "scaler_state" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state"])
            _restore_rng_state(ckpt["rng_state"])
            step = int(ckpt["step"])
            epoch = int(ckpt["epoch"])
            batch_offset = int(ckpt["batch_offset"])
            examples_seen = int(ckpt["examples_seen"])
            best_score = float(ckpt.get("best_score", -1.0))
            history = list(ckpt.get("history", []))
            last_val_metrics = dict(ckpt.get("val_metrics", {}))
            print(f"Resumed verified checkpoint at step={step}/{max_updates} (epoch={epoch})", flush=True)

    stop_requested = {"flag": False, "reason": ""}

    def _sig_handler(signum: int, _frame: Any) -> None:
        stop_requested["flag"] = True
        stop_requested["reason"] = f"signal_{signum}"

    prev_sigint = signal.signal(signal.SIGINT, _sig_handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _sig_handler)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_train_start = time.perf_counter()
    start_step = step
    start_examples = examples_seen
    last_saved_step = step

    def persist_checkpoint(status_label: str, val_m: dict[str, Any] | None = None, is_best: bool = False) -> dict[str, Any]:
        nonlocal last_saved_step
        elapsed_now = max(1e-6, time.perf_counter() - t_train_start)
        delta_steps = step - start_step
        delta_ex = examples_seen - start_examples
        upd_per_s = delta_steps / elapsed_now
        win_per_s = delta_ex / elapsed_now
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else 0.0
        est_total_min = round((max_updates / upd_per_s) / 60.0, 2) if upd_per_s > 0 else None

        payload = {
            "job_id": job_id,
            "scale": args.scale,
            "seed": seed,
            "window_s": window_s,
            "window_samples": data_meta["window_samples"],
            "n_classes": int(arch["n_classes"]),
            "step": step,
            "max_updates": max_updates,
            "epoch": epoch,
            "batch_offset": batch_offset,
            "examples_seen": examples_seen,
            "best_score": best_score,
            "val_metrics": val_m or last_val_metrics,
            "history": history,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": _rng_state_dict(),
            "config_sha256": cfg_hash,
            "manifest_sha256": manifest_hash,
        }
        latest_sha = _atomic_save_ckpt(ckpt_latest, payload)
        best_sha = _atomic_save_ckpt(ckpt_best, payload) if is_best else (sha256_file(ckpt_best) if ckpt_best.is_file() else None)
        last_saved_step = step

        drive_verified = False
        if drive_job_dir is not None:
            copy_with_sha256_verify(ckpt_latest, drive_job_dir / "ckpt_latest.pt")
            if ckpt_best.is_file():
                copy_with_sha256_verify(ckpt_best, drive_job_dir / "ckpt_best.pt")
            drive_verified = True

        job_state = {
            "job_id": job_id,
            "status": status_label,
            "scale": args.scale,
            "seed": seed,
            "window_s": window_s,
            "step": step,
            "max_updates": max_updates,
            "epoch": epoch,
            "examples_seen": examples_seen,
            "best_score": best_score,
            "val_metrics": val_m or last_val_metrics,
            "data_meta": data_meta,
            "runtime": {
                "device_name": profile["device_name"],
                "vram_gib": profile["vram_gib"],
                "amp_mode": amp_mode,
                "batch_size": batch_size,
                "data_load_s": data_load_s,
                "windows_per_sec": round(win_per_s, 1),
                "updates_per_sec": round(upd_per_s, 2),
                "peak_vram_mb": peak_vram_mb,
                "estimated_full_run_min": est_total_min,
            },
            "ckpt_latest_sha256": latest_sha,
            "ckpt_best_sha256": best_sha,
            "drive_verified": drive_verified,
            "history": history,
        }
        write_json(state_path, job_state)
        if drive_job_dir is not None:
            copy_with_sha256_verify(state_path, drive_job_dir / "job_state.json")
        _sync_ledger(work_root, drive_root, job_id, job_state)
        return job_state

    try:
        n_train = len(X_train)
        amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16 if amp_mode == "fp16" else None
        target_stop = min(max_updates, step + args.stop_after_updates) if args.stop_after_updates is not None else max_updates

        while step < target_stop and not stop_requested["flag"]:
            gen = torch.Generator().manual_seed(seed + epoch)
            perm = torch.randperm(n_train, generator=gen).numpy()
            batches = [perm[i : i + batch_size] for i in range(0, n_train, batch_size)]
            for b_idx in range(batch_offset, len(batches)):
                if step >= target_stop or stop_requested["flag"]:
                    break
                idx = batches[b_idx]
                xb = torch.from_numpy(X_train[idx]).to(device, non_blocking=True)
                yb = torch.from_numpy(y_train[idx]).to(device, non_blocking=True)

                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss encountered at step {step + 1}.")

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.constrain_weights()

                step += 1
                batch_offset = b_idx + 1
                examples_seen += len(idx)

                need_eval = (step % eval_every == 0) or (step == max_updates) or (step == target_stop)
                need_ckpt = (step % ckpt_every == 0) or (step == max_updates) or (step == target_stop) or stop_requested["flag"]

                if need_eval:
                    last_val_metrics = evaluate_model(model, X_val, y_val, class_weights, device)
                    last_val_metrics["step"] = step
                    last_val_metrics["train_loss"] = round(float(loss.item()), 5)
                    last_val_metrics["lr"] = round(float(optimizer.param_groups[0]["lr"]), 7)
                    history.append(last_val_metrics)
                    score_cur = float(last_val_metrics["val_auc_macro_mi"] or last_val_metrics["val_bal_acc_3class"])
                    is_best = score_cur > best_score
                    if is_best:
                        best_score = score_cur
                    print(
                        f"[step {step}/{max_updates}] loss={last_val_metrics['train_loss']:.4f} "
                        f"val_loss={last_val_metrics['val_loss']:.4f} "
                        f"bal_acc3={last_val_metrics['val_bal_acc_3class']:.4f} "
                        f"auc_mi_idle={last_val_metrics['val_auc_mi_vs_clean_idle']} "
                        f"auc_mi_relax={last_val_metrics['val_auc_mi_vs_deliberate_relax']}",
                        flush=True,
                    )
                else:
                    is_best = False

                if need_ckpt:
                    st_lbl = "completed" if step >= max_updates else "in_progress"
                    persist_checkpoint(st_lbl, last_val_metrics, is_best=is_best)

            if batch_offset >= len(batches):
                epoch += 1
                batch_offset = 0

        if stop_requested["flag"] and step > last_saved_step:
            last_val_metrics = evaluate_model(model, X_val, y_val, class_weights, device)
            last_val_metrics["step"] = step
            persist_checkpoint("interrupted", last_val_metrics, is_best=False)
            print(f"Interrupted cleanly via {stop_requested['reason']} at step {step}.", flush=True)
            return 130

        if not last_val_metrics or step != last_saved_step:
            last_val_metrics = evaluate_model(model, X_val, y_val, class_weights, device)
            last_val_metrics["step"] = step
            final_lbl = "completed" if step >= max_updates else "in_progress"
            persist_checkpoint(final_lbl, last_val_metrics, is_best=False)

        final_state = json.loads(state_path.read_text(encoding="utf-8"))
        print(
            f"Finished phase step={step}/{max_updates} ({final_state['status']}) | "
            f"throughput={final_state['runtime']['windows_per_sec']} win/s | "
            f"drive_verified={final_state['drive_verified']}",
            flush=True,
        )
        return 0
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)


def _read_ledger(work_root: Path, drive_root: Path | None) -> dict[str, Any]:
    for root in [p for p in (drive_root, work_root) if p is not None]:
        lp = Path(root) / "campaign_ledger.json"
        if lp.is_file():
            return json.loads(lp.read_text(encoding="utf-8"))
    return {"schema_version": "phase2-campaign-ledger-v1", "jobs": {}}


def cmd_next_job(args: argparse.Namespace) -> int:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    schedule = build_campaign_schedule(cfg)
    if args.scale:
        schedule = [j for j in schedule if j["scale"] == args.scale]
    ledger = _read_ledger(Path(args.work_dir), Path(args.drive_dir) if args.drive_dir else None)
    jobs_state = ledger.get("jobs", {})

    resumable, pending, completed = [], [], []
    for job in schedule:
        st = jobs_state.get(job["job_id"])
        if st and st.get("status") == "completed" and int(st.get("step", 0)) >= job["max_updates"]:
            completed.append(job)
        elif st and int(st.get("step", 0)) > 0:
            resumable.append({**job, "resume_step": int(st["step"]), "status": st.get("status")})
        else:
            pending.append({**job, "resume_step": 0, "status": "pending"})

    chosen = resumable[0] if resumable else (pending[0] if pending else None)
    summary = {
        "completed_count": len(completed),
        "total_count": len(schedule),
        "next_job": chosen,
        "completed_jobs": [j["job_id"] for j in completed],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.execute and chosen is not None:
        run_args = argparse.Namespace(
            config=args.config,
            split_config=args.split_config,
            manifest=args.manifest,
            shard_dir=args.shard_dir,
            work_dir=args.work_dir,
            drive_dir=args.drive_dir,
            scale=chosen["scale"],
            seed=chosen["seed"],
            window_s=chosen["window_s"],
            max_updates=chosen["max_updates"],
            checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            job_id=chosen["job_id"],
            resume=chosen["resume_step"] > 0,
            stop_after_updates=args.stop_after_updates,
            train_shards=None,
            val_shards=None,
        )
        return cmd_train(run_args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 Stieger causal ingestion and resumable GPU training.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Stage verified causal 100 Hz Stieger shards.")
    p_ing.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_ing.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CFG)
    p_ing.add_argument("--shard-dir", type=Path, required=True)
    p_ing.add_argument("--raw-dir", type=Path, default=None)
    p_ing.add_argument("--drive-shard-dir", type=Path, default=None)
    p_ing.add_argument("--scale", choices=["scale_10pct", "scale_30pct", "scale_100pct"], default=None)
    p_ing.add_argument("--include-dev", action="store_true")
    p_ing.add_argument("--sessions", nargs="+", default=None)
    p_ing.add_argument("--max-sessions", type=int, default=None)
    p_ing.add_argument("--keep-mat", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    p_tr = sub.add_parser("train", help="Run resumable causal 3-class EEGNet pretraining.")
    p_tr.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CFG)
    p_tr.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CFG)
    p_tr.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p_tr.add_argument("--shard-dir", type=Path, required=True)
    p_tr.add_argument("--work-dir", type=Path, required=True)
    p_tr.add_argument("--drive-dir", type=Path, default=None)
    p_tr.add_argument("--scale", choices=["scale_10pct", "scale_30pct", "scale_100pct"], default=None)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--window-s", type=float, default=None)
    p_tr.add_argument("--max-updates", type=int, default=None)
    p_tr.add_argument("--checkpoint-every", type=int, default=None)
    p_tr.add_argument("--eval-every", type=int, default=None)
    p_tr.add_argument("--batch-size", type=int, default=None)
    p_tr.add_argument("--lr", type=float, default=None)
    p_tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p_tr.add_argument("--job-id", type=str, default=None)
    p_tr.add_argument("--resume", action="store_true")
    p_tr.add_argument("--stop-after-updates", type=int, default=None)
    p_tr.add_argument("--train-shards", nargs="+", default=None)
    p_tr.add_argument("--val-shards", nargs="+", default=None)
    p_tr.set_defaults(func=cmd_train)

    for cmd_name in ("status", "next-job"):
        p_nx = sub.add_parser(cmd_name, help="Inspect or launch next incomplete Phase 2 job.")
        p_nx.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CFG)
        p_nx.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CFG)
        p_nx.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        p_nx.add_argument("--shard-dir", type=Path, default=Path("outputs/phase2_stieger_cache/shards"))
        p_nx.add_argument("--work-dir", type=Path, default=Path("outputs/phase2_work"))
        p_nx.add_argument("--drive-dir", type=Path, default=None)
        p_nx.add_argument("--scale", choices=["scale_10pct", "scale_30pct", "scale_100pct"], default=None)
        p_nx.add_argument("--checkpoint-every", type=int, default=None)
        p_nx.add_argument("--eval-every", type=int, default=None)
        p_nx.add_argument("--batch-size", type=int, default=None)
        p_nx.add_argument("--lr", type=float, default=None)
        p_nx.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p_nx.add_argument("--stop-after-updates", type=int, default=None)
        p_nx.add_argument("--execute", action="store_true")
        p_nx.set_defaults(func=cmd_next_job)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
