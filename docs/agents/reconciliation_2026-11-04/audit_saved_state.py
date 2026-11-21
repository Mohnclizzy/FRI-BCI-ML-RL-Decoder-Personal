"""Read saved project evidence without launching models or rollouts.
Usage: python3 audit_saved_state.py OUTPUT_JSON (new path only).
"""
import datetime, hashlib, json, pathlib, sys
root = pathlib.Path("/home/ubuntu")
repo = root / "FRI-BCI-ML-RL-Decoder-Personal"
raw = root / "robomimic-native-phase6-20261104/matched_batch"
runs = []
for path in sorted(raw.glob("*/summary.json")):
    summary = json.loads(path.read_text())
    trace = path.with_name("telemetry.jsonl")
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    latch = summary["latch_step"]
    baseline = rows[0]["cube_pos"][2]
    loss = [r["step"] for r in rows[latch:] if not r["bilateral_contact"] or r["cube_pos"][2] - baseline < .08] if latch is not None else []
    locked = [r for r in rows if r["mode"] in ("hold", "latch")]
    oldpath = root / "robomimic-native-phase5-20261104/matched_batch" / path.parent.name / "telemetry.jsonl"
    old = json.loads(oldpath.open().readline())
    checks = {
        "401_sequential_states": len(rows) == 401 and [r["step"] for r in rows] == list(range(401)),
        "20hz_clock": all(abs(r["time_s"] - r["step"] / 20) < 1e-9 for r in rows),
        "contact_agrees_with_builtin": all(r["bilateral_contact"] == r["builtin_grasp"] for r in rows),
        "retention_matches_trace": summary["terminal_retained"] == bool(latch is not None and not loss),
        "summary_hold_and_latch_invariants": summary["hold_invariants_pass"] and summary["latch_invariants_pass"],
        "no_policy_or_rnn_advance_when_locked": all(r["policy_calls"] == rows[r["step"]-1]["policy_calls"] and r["rnn_counter"] == rows[r["step"]-1]["rnn_counter"] for r in locked),
        "fixed_latch_pose_and_gripper": all(r["goal_pos"] == summary["latch_goal_pos"] and r["goal_ori"] == summary["latch_goal_ori"] and r["gripper_command"] == [-1, 1] for r in rows if r["mode"] == "latch"),
        "phase5_initial_cube_and_eef_match": old["cube_pos"] == rows[0]["cube_pos"] and old["eef_pos"] == rows[0]["eef_pos"],
    }
    runs.append({"run": path.parent.name, "acquired": latch is not None, "retained": summary["terminal_retained"], "terminal_rise_m": summary["terminal_rise_m"], "checks": checks, "trace_path": str(trace), "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()})
sessions = []
for home in (root / ".codex", root / ".codex-cli-runtime"):
    for p in sorted((home / "sessions").rglob("*.jsonl")):
        if "/2026/11/03/" not in str(p) and "/2026/11/04/" not in str(p):
            continue
        meta = {}
        for line in p.open():
            event = json.loads(line)
            if event.get("type") == "session_meta":
                meta = {k: event["payload"].get(k) for k in ("id", "cwd", "originator", "cli_version")}
                break
        sessions.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "metadata": meta})
sources = {}
for p in sorted((repo / "results/robomimic_latch6_zoffset_2026-11-04").glob("*")):
    if p.is_file():
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
report = {
    "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Read-only saved-artifact consistency audit; not full protocol qualification.",
    "runs": runs,
    "positive_retained": sum(r["retained"] for r in runs if not r["run"].startswith("always-hold")),
    "negative_acquired": sum(r["acquired"] for r in runs if r["run"].startswith("always-hold")),
    "all_listed_checks_pass": len(runs) == 24 and all(all(r["checks"].values()) for r in runs),
    "phase6_source_hashes": sources,
    "phase6_protocol_paths_observed": [str(p) for b in (raw.parent, repo / "results/robomimic_latch6_zoffset_2026-11-04") for p in b.rglob("*protocol*")],
    "recent_codex_session_inventory": sessions,
    "limitations": ["Contact sampled at 20 Hz, not every physics substep.", "No independent held-out qualification.", "HOLD drift values are checked via saved summary flags, not independently recomputed here.", "Initial cube and end-effector equality is not proof of complete simulator-state equality.", "No off-host recovery check for Phase 6 in this audit.", "No raw chat messages or credentials exported."]
}
with pathlib.Path(sys.argv[1]).open("x") as f:
    json.dump(report, f, indent=2); f.write("\n")
print(json.dumps({"output": sys.argv[1], "runs": len(runs), "positive_retained": report["positive_retained"], "all_listed_checks_pass": report["all_listed_checks_pass"], "sessions_indexed": len(sessions)}))

