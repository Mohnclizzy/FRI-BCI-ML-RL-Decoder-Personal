"""Work Package 1: Stieger Figshare manifest, pre-inspection deterministic split, and 3-session audit.

Downloads one dev session .mat at a time into gitignored cache, verifies Figshare MD5/size,
audits structure/continuity/labels/timing, writes strictly causal 100 Hz 9-channel .npz shard,
and immediately deletes raw .mat to keep local disk usage bounded.
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eeg_decoder.stieger import (  # noqa: E402
    attach_split_to_manifest,
    build_deterministic_split,
    download_figshare_file,
    evaluate_dev_shard_separability,
    fetch_figshare_article,
    inspect_and_convert_mat,
    parse_figshare_manifest,
    sha256_file,
    write_json,
)

DEFAULT_AUDIT_SESSIONS = ["S1_Session_1", "S1_Session_7", "S2_Session_1"]
RESULTS_DIR = REPO_ROOT / "results" / "phase2_stieger_audit_2026-11-19"
SPLIT_CONFIG_PATH = REPO_ROOT / "configs" / "phase2_stieger_split.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "outputs" / "phase2_stieger_cache"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stieger WP1 metadata split and 3-session causal audit.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--split-config", type=Path, default=SPLIT_CONFIG_PATH)
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_AUDIT_SESSIONS)
    parser.add_argument("--keep-mat", action="store_true", help="Retain raw .mat after shard conversion (off by default).")
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    cache_raw = args.cache_dir / "raw"
    cache_shards = args.cache_dir / "shards"
    cache_raw.mkdir(parents=True, exist_ok=True)
    cache_shards.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Step A: Build & freeze deterministic participant split BEFORE downloading any .mat signal
    print("Fetching Figshare article 13123148 metadata...")
    raw_article = fetch_figshare_article()
    raw_manifest = parse_figshare_manifest(raw_article)
    split_cfg = build_deterministic_split(raw_manifest, seed=args.seed, pinned_dev_subjects=(1, 2), n_dev=8, n_test=10)
    manifest = attach_split_to_manifest(raw_manifest, split_cfg)

    write_json(args.split_config, split_cfg)
    manifest_path = args.out_dir / "stieger_figshare_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Froze split before inspection: {args.split_config}")
    print(
        f"Split counts -> source_train: {split_cfg['counts']['source_100pct']['n_subjects']} subj "
        f"({split_cfg['counts']['source_100pct']['n_sessions']} sess, {split_cfg['counts']['source_100pct']['total_gib']} GiB), "
        f"dev: {split_cfg['counts']['dev']['n_subjects']} subj ({split_cfg['counts']['dev']['n_sessions']} sess), "
        f"test: {split_cfg['counts']['test']['n_subjects']} subj ({split_cfg['counts']['test']['n_sessions']} sess)"
    )

    by_key = {row["session_key"]: row for row in manifest["sessions"]}
    dev_set = set(split_cfg["dev_subjects"])
    for key in args.sessions:
        if key not in by_key:
            raise KeyError(f"Unknown session key: {key}")
        if int(by_key[key]["subject_id"]) not in dev_set:
            raise PermissionError(f"Refusing to audit non-dev session {key}; dev_subjects={sorted(dev_set)}")

    session_reports = []
    local_shard_manifest = {
        "schema_version": "phase2-stieger-shards-v1",
        "source_manifest_sha256": sha256_file(manifest_path),
        "split_config_sha256": sha256_file(args.split_config),
        "shards": {},
    }

    for key in args.sessions:
        meta = by_key[key]
        free_gib = shutil.disk_usage(args.cache_dir).free / (1024**3)
        print(f"\n[{key}] Free disk: {free_gib:.2f} GiB; downloading {meta['size_bytes'] / 1e6:.1f} MB...")
        if free_gib < 1.2:
            raise RuntimeError(f"Insufficient disk space ({free_gib:.2f} GiB free) to stage {key}.")

        mat_path = cache_raw / meta["filename"]
        dl_info = download_figshare_file(
            url=meta["download_url"],
            dest_path=mat_path,
            expected_size=meta["size_bytes"],
            expected_md5=meta["computed_md5"],
        )
        print(f"[{key}] Downloaded & verified MD5={dl_info['md5']}, SHA256={dl_info['sha256'][:16]}...")

        shard_path = cache_shards / f"{key}.npz"
        report = inspect_and_convert_mat(
            mat_path=mat_path,
            shard_path=shard_path,
            subject_id=meta["subject_id"],
            session_id=meta["session_id"],
            source_md5=dl_info["md5"],
            source_sha256=dl_info["sha256"],
        )
        gc.collect()
        if not args.keep_mat and mat_path.is_file():
            mat_path.unlink()
            print(f"[{key}] Deleted raw .mat; kept shard {shard_path.name} ({report['shard_bytes'] / 1e6:.2f} MB)")

        auc_2s = evaluate_dev_shard_separability(shard_path, window_s=2.0)
        auc_1s = evaluate_dev_shard_separability(shard_path, window_s=1.0)
        report["figshare_file_id"] = meta["file_id"]
        report["figshare_url"] = meta["download_url"]
        report["source_bytes"] = meta["size_bytes"]
        report["source_md5"] = dl_info["md5"]
        report["source_sha256"] = dl_info["sha256"]
        report["separability_2s"] = auc_2s
        report["separability_1s"] = auc_1s
        session_reports.append(report)

        local_shard_manifest["shards"][key] = {
            "session_key": key,
            "subject_id": meta["subject_id"],
            "session_id": meta["session_id"],
            "partition": meta["partition"],
            "shard_path": str(shard_path),
            "shard_bytes": report["shard_bytes"],
            "shard_sha256": report["shard_sha256"],
            "source_md5": dl_info["md5"],
            "source_sha256": dl_info["sha256"],
        }

    write_json(args.cache_dir / "shard_manifest.json", local_shard_manifest)

    mean_shard_bytes = sum(r["shard_bytes"] for r in session_reports) / len(session_reports)
    mean_raw_bytes = sum(r["source_bytes"] for r in session_reports) / len(session_reports)
    compression_ratio = mean_shard_bytes / mean_raw_bytes
    projected_storage = {
        "measured_mean_raw_mb": round(mean_raw_bytes / 1e6, 2),
        "measured_mean_100hz_9ch_shard_mb": round(mean_shard_bytes / 1e6, 2),
        "shard_to_raw_ratio": round(compression_ratio, 4),
        "projected_shards_gib_by_scale": {
            "scale_10pct": round(split_cfg["counts"]["source_10pct"]["n_sessions"] * mean_shard_bytes / (1024**3), 3),
            "scale_30pct": round(split_cfg["counts"]["source_30pct"]["n_sessions"] * mean_shard_bytes / (1024**3), 3),
            "scale_100pct": round(split_cfg["counts"]["source_100pct"]["n_sessions"] * mean_shard_bytes / (1024**3), 3),
            "all_598_sessions": round(manifest["n_sessions"] * mean_shard_bytes / (1024**3), 3),
        },
        "projected_total_raw_gib": round(manifest["total_article_bytes"] / (1024**3), 2),
    }

    audit_metrics = {
        "schema_version": "phase2-stieger-audit-v1",
        "split_config_path": str(args.split_config.relative_to(REPO_ROOT)).replace("\\", "/"),
        "split_config_sha256": sha256_file(args.split_config),
        "manifest_sha256": sha256_file(manifest_path),
        "audited_session_keys": args.sessions,
        "corpus_summary": {
            "n_subjects": manifest["n_subjects"],
            "n_sessions": manifest["n_sessions"],
            "total_article_bytes": manifest["total_article_bytes"],
            "total_raw_gib": round(manifest["total_article_bytes"] / (1024**3), 2),
            "split_counts": split_cfg["counts"],
        },
        "projected_storage": projected_storage,
        "sessions": session_reports,
    }
    audit_path = args.out_dir / "audit_metrics.json"
    write_json(audit_path, audit_metrics)
    print(f"\nSaved audit metrics: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
