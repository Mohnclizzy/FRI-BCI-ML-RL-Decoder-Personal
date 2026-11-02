"""Three command baselines on identical continuous estimates; no robot claims."""

from dataclasses import asdict
import json
from pathlib import Path

from .commands import CommandGate, Estimate, Skill


def compare_trace(path: Path, output: Path, threshold=0.7, persistence_s=0.5, max_age_s=1.0):
    gates = {mode: CommandGate(threshold, persistence_s, max_age_s, mode=mode)
             for mode in ("raw", "confidence", "temporal")}
    counts = {mode: {"estimates": 0, "execute_estimates": 0, "command_switches": 0}
              for mode in gates}
    previous, last_key = {}, None
    output.parent.mkdir(parents=True, exist_ok=True)
    with path.open(encoding="utf-8") as incoming, output.open("x", encoding="utf-8") as outgoing:
        for line in incoming:
            if not line.strip():
                continue
            estimate = Estimate(**json.loads(line)["estimate"])
            key = (estimate.source, estimate.episode)
            if key != last_key:
                previous, last_key = {}, key
            decisions = {}
            for mode, gate in gates.items():
                request = gate.update(estimate, estimate.available_at_s)
                counts[mode]["estimates"] += 1
                counts[mode]["execute_estimates"] += request.skill == Skill.GRASP_LIFT
                counts[mode]["command_switches"] += mode in previous and previous[mode] != request.skill
                previous[mode] = request.skill
                decisions[mode] = asdict(request)
            outgoing.write(json.dumps({"estimate": asdict(estimate), "decisions": decisions}) + "\n")
    if not counts["raw"]["estimates"]:
        raise ValueError("No estimates in input trace")
    return {"scope": "command decisions only; not physical success, intent accuracy, or live latency",
            "threshold": threshold, "persistence_s": persistence_s, "max_age_s": max_age_s,
            "conditions": counts}
