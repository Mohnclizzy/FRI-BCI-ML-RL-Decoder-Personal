"""Execute an immutable bounded_study robot_jobs.json with fail-fast resumability."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    manifest = json.loads((work / "robot_jobs.json").read_text())
    jobs = manifest["jobs"]
    if len(jobs) != 270 or len({(j["subject"], j["seed"], j["condition"]) for j in jobs}) != len(jobs):
        raise ValueError("Expected 270 unique frozen jobs")
    root = work / "robot_results"
    root.mkdir(exist_ok=True)
    status_path = root / "progress.json"
    done = set()
    if status_path.exists():
        state = json.loads(status_path.read_text())
        if state["manifest_sha256"] != __import__("hashlib").sha256((work / "robot_jobs.json").read_bytes()).hexdigest():
            raise ValueError("Job manifest changed since previous run")
        done = set(state["completed"])
    completed = list(sorted(done))
    manifest_sha = __import__("hashlib").sha256((work / "robot_jobs.json").read_bytes()).hexdigest()
    for i, job in enumerate(jobs):
        key = "%s_seed%d_%s" % ("S%03d" % job["subject"], job["seed"], job["condition"])
        if key in done:
            continue
        out = root / key
        cmd = [sys.executable, str(args.harness), "--seed", str(job["seed"]), "--out", str(out),
               "--condition", job["harness_condition"], "--ckpt", str(args.checkpoint)]
        if job["harness_condition"] == "eeg":
            cmd += ["--gate", "raw", "--decisions", str(work / job["trace"].replace("\\", "/"))]
        elif job["harness_condition"] == "enable-schedule":
            cmd += ["--schedule-json", str(work / job["schedule"].replace("\\", "/"))]
        if args.video and job["subject"] == 1 and job["seed"] == 0 and job["condition"] == "selected":
            cmd.append("--video")
        started = time.time()
        with (root / (key + ".log")).open("w") as log:
            result = subprocess.run(cmd, cwd=str(work), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Job failed; stopped at %s (exit %d)" % (key, result.returncode))
        summary = json.loads((out / "summary.json").read_text())
        if summary["harness_version"] != "robomimic-latch6-v2.0.0" or summary["seed"] != job["seed"]:
            raise ValueError("Output provenance mismatch: " + key)
        completed.append(key)
        (root / key / "job.json").write_text(json.dumps(dict(job=job, elapsed_s=time.time()-started,
            manifest_sha256=manifest_sha), indent=2) + "\n")
        status_path.write_text(json.dumps(dict(manifest_sha256=manifest_sha, completed=completed,
            last=key, count=len(completed), total=len(jobs)), indent=2) + "\n")
        print("%d/%d %s retained=%s policy_calls=%d" % (len(completed),len(jobs),key,
            summary["terminal_retained"],summary["policy_calls"]), flush=True)
    print("All frozen jobs complete", flush=True)


if __name__ == "__main__":
    main()
