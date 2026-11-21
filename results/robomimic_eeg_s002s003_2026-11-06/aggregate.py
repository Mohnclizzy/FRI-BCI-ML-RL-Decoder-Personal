#!/usr/bin/env python
"""Aggregate the S002/S003 EEG-arm held-out rollouts (2 subjects x 3 gates x 3 seeds).

Reads each <subject>_<gate>_seed<n>/summary.json and produces:
  - eeg_arm.tsv        : one row per rollout (subject, gate, seed, completion_state, ...)
  - eeg_arm_summary.md : per-(subject, gate) aggregate table + cross-subject roll-up
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def label(s):
    if s.get("terminal_retained"):
        return "complete_retained"
    if s.get("first_acquisition_step") is None:
        return "contact_none"
    if s.get("post_acquisition_loss"):
        return "contact_lost"
    return "complete_unretained"


def find_summaries():
    rows = []
    for d in sorted(BASE.iterdir()):
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if s.get("condition") != "eeg":
            continue
        name = d.name  # e.g. S002_raw_seed0
        subj, gate, seedtok = name.split("_")
        rows.append({
            "subject": subj, "gate": gate, "seed": int(seedtok[-1]),
            "completion_state": label(s),
            "terminal_retained": bool(s.get("terminal_retained")),
            "post_acquisition_loss": bool(s.get("post_acquisition_loss")),
            "first_acquisition_step": s.get("first_acquisition_step"),
            "max_rise_m": s.get("max_rise_m"),
            "native_success_any": bool(s.get("native_success_any")),
            "latch_invariants_pass": bool(s.get("latch_invariants_pass")),
            "hold_invariants_pass": bool(s.get("hold_invariants_pass")),
            "checkpoint_sha256": s.get("checkpoint_sha256"),
            "decisions_sha256": s.get("decisions_sha256"),
        })
    rows.sort(key=lambda r: (r["subject"], r["gate"], r["seed"]))
    return rows


def main():
    rows = find_summaries()
    if not rows:
        raise SystemExit("No eeg rollouts found")

    cols = ["subject", "gate", "seed", "completion_state", "terminal_retained",
            "post_acquisition_loss", "first_acquisition_step", "max_rise_m"]
    with (BASE / "eeg_arm.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # per-(subject, gate) aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["subject"], r["gate"])].append(r)

    print(f"{'subject':8} {'gate':11} {'retained':9} {'first_acq':>9} {'max_rise':>9}  seeds")
    for (subj, gate) in sorted(agg):
        rs = agg[(subj, gate)]
        retained = sum(r["terminal_retained"] for r in rs)
        acq = [r["first_acquisition_step"] for r in rs if r["first_acquisition_step"] is not None]
        acq_s = "/".join(str(x) for x in acq) if acq else "none"
        rise = max((r["max_rise_m"] or 0.0) for r in rs)
        print(f"{subj:8} {gate:11} {retained}/{len(rs)}    {acq_s:>9} {rise:9.4f}  seeds={[r['seed'] for r in rs]}")

    # write the markdown summary table (data block for the doc)
    lines = ["| subject | gate | retained | first_acq_step (s0/s1/s2) | max_rise_m |",
             "|---|---|---|---|---|"]
    for (subj, gate) in sorted(agg):
        rs = agg[(subj, gate)]
        retained = sum(r["terminal_retained"] for r in rs)
        acq = [r["first_acquisition_step"] for r in rs if r["first_acquisition_step"] is not None]
        acq_s = "/".join(str(x) for x in sorted(acq)) if acq else "none"
        rise = max((r["max_rise_m"] or 0.0) for r in rs)
        lines.append(f"| {subj} | {gate} | {retained}/{len(rs)} | {acq_s} | {rise:.4f} |")
    (BASE / "eeg_arm_agg.md").write_text("\n".join(lines) + "\n")
    print("\nwrote eeg_arm.tsv and eeg_arm_agg.md")
    print("checkpoint:", rows[0]["checkpoint_sha256"])
    print("S002 decisions_sha:", rows[0]["decisions_sha256"] if rows[0]["subject"]=="S002" else "n/a")


if __name__ == "__main__":
    main()
