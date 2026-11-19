"""Execute all 240 (seed in 0..2, window_start in 1,6,..,396) one-request-starts-one-lift
rollouts through the versioned simulation.src.fri_bci_sim.robomimic_harness.run_rollout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.src.fri_bci_sim import robomimic_harness as harness

START_STEPS = list(range(1, 400, 5))  # 80 4Hz start steps: 1, 6, 11, ..., 396
SEEDS = (0, 1, 2)
EXPECTED_CKPT_SHA256 = "3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605"


def _run_worker(args_tuple: tuple[int, list[int], str, str]) -> list[dict]:
    seed, steps, ckpt_str, work_root_str = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    work_root = Path(work_root_str)
    out_summaries = []
    for s_start in steps:
        window_steps = 400 - s_start + 1
        job_key = f"seed{seed}_start{s_start:03d}"
        job_dir = work_root / job_key
        if job_dir.exists():
            shutil.rmtree(job_dir)
        argv = [
            "--seed", str(seed),
            "--out", str(job_dir),
            "--condition", "enable-window",
            "--window-start", str(s_start),
            "--window-steps", str(window_steps),
            "--ckpt", ckpt_str,
        ]
        # Redirect stdout during batch so 240 full JSON dumps don't flood stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            summary = harness.run_rollout(argv)
        finally:
            sys.stdout = old_stdout

        summary_path = job_dir / "summary.json"
        telemetry_path = job_dir / "telemetry.jsonl"
        if not summary_path.is_file() or not telemetry_path.is_file():
            raise RuntimeError(f"Missing harness output files for {job_key}")
        summary["telemetry_sha256"] = hashlib.sha256(telemetry_path.read_bytes()).hexdigest()
        summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        out_summaries.append(summary)
    return out_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT / "artifacts" / "lift_ph_low_dim_epoch_1000_succ_100.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for robomimic_240_summaries.json and robomimic_240_rollouts_bundle.tar.gz",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/robomimic_one_request_240"),
        help="Temporary directory for per-job summary.json and telemetry.jsonl before archiving",
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    ckpt = args.ckpt.resolve()
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if ckpt_sha != EXPECTED_CKPT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {ckpt_sha} != {EXPECTED_CKPT_SHA256}")

    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Split the 240 jobs across workers by chunking start steps per seed
    tasks = []
    chunks_per_seed = max(1, args.workers // len(SEEDS))
    for seed in SEEDS:
        for i in range(chunks_per_seed):
            chunk = START_STEPS[i::chunks_per_seed]
            if chunk:
                tasks.append((seed, chunk, str(ckpt), str(work_dir)))

    t0 = time.time()
    all_summaries: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        for batch in ex.map(_run_worker, tasks):
            all_summaries.extend(batch)
    elapsed_s = time.time() - t0

    if len(all_summaries) != 240:
        raise RuntimeError(f"Expected 240 rollout summaries, got {len(all_summaries)}")

    by_seed: dict[str, dict[str, dict]] = {"0": {}, "1": {}, "2": {}}
    for s in sorted(all_summaries, key=lambda r: (r["seed"], r["window_start"])):
        if s["harness_version"] != "robomimic-latch6-v2.0.0":
            raise ValueError(f"Unexpected harness_version: {s['harness_version']}")
        if not s["hold_invariants_pass"] or not s["latch_invariants_pass"] or not s["contact_proxy_validated"]:
            raise ValueError(f"Invariant violation in seed={s['seed']} window_start={s['window_start']}")
        by_seed[str(s["seed"])][str(s["window_start"])] = s

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out_dir / "robomimic_240_rollouts_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for seed in SEEDS:
            for s_start in START_STEPS:
                job_key = f"seed{seed}_start{s_start:03d}"
                tar.add(work_dir / job_key, arcname=job_key)

    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    payload = {
        "harness_version": harness.HARNESS_VERSION,
        "harness_sha256": harness.harness_source_sha256(),
        "checkpoint_sha256": ckpt_sha,
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle_sha,
        "total_rollouts": len(all_summaries),
        "elapsed_s": round(elapsed_s, 2),
        "all_hold_invariants_pass": all(s["hold_invariants_pass"] for s in all_summaries),
        "all_latch_invariants_pass": all(s["latch_invariants_pass"] for s in all_summaries),
        "all_contact_proxy_validated": all(s["contact_proxy_validated"] for s in all_summaries),
        "any_post_acquisition_loss": any(s["post_acquisition_loss"] for s in all_summaries),
        "rollouts_by_seed": by_seed,
    }
    summaries_path = args.out_dir / "robomimic_240_summaries.json"
    summaries_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Completed 240 versioned-harness rollouts in {elapsed_s:.1f}s")
    print(f"all_hold_invariants_pass={payload['all_hold_invariants_pass']} "
          f"all_latch_invariants_pass={payload['all_latch_invariants_pass']} "
          f"all_contact_proxy_validated={payload['all_contact_proxy_validated']} "
          f"any_post_acquisition_loss={payload['any_post_acquisition_loss']}")
    for seed in ("0", "1", "2"):
        ret_steps = [int(k) for k, v in by_seed[seed].items() if v["terminal_retained"]]
        losses = [int(k) for k, v in by_seed[seed].items() if v["post_acquisition_loss"]]
        print(f"Seed {seed}: retained={len(ret_steps)}/80 (max_start={max(ret_steps)}), post_acquisition_loss={len(losses)}")


if __name__ == "__main__":
    main()
