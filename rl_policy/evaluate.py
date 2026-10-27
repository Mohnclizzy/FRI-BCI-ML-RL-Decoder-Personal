"""Replay one held-out EEG prediction per simulated episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from rl_policy import HAND_IMAGERY, REST
from rl_policy.env import GraspHoldEnv


def load_predictions(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prediction file must be a nonempty JSON list")
    seen = set()
    trials = []
    for row in rows:
        trial = row["trial_id"]
        if trial in seen:
            raise ValueError(f"Duplicate trial ID: {trial}")
        seen.add(trial)
        true = int(row["label_id"])
        decoded = {"REST": REST, "HAND_IMAGERY": HAND_IMAGERY}[row["predicted_intent"]]
        if true not in (REST, HAND_IMAGERY):
            raise ValueError(f"Unknown label ID on {trial}")
        trials.append((trial, true, decoded))
    return trials


def run_episode(env, policy, true_intent, command):
    observation, _ = env.reset(options={"true_intent": true_intent, "command": command})
    while True:
        if policy == "always_hold":
            action = 0
        elif policy == "always_grasp":
            action = 1 if not observation[2] else 2
        else:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info


def measure(env, trials, policy, use_oracle=False):
    results = []
    for trial_id, true, decoded in trials:
        command = true if use_oracle else decoded
        info = run_episode(env, policy, true, command)
        results.append({"trial_id": trial_id, "true_intent": true, "command": command,
                        "success": info["success"], "lifted": info["lifted"]})
    rest = [row for row in results if row["true_intent"] == REST]
    imagery = [row for row in results if row["true_intent"] == HAND_IMAGERY]
    if not rest or not imagery:
        raise ValueError("Evaluation requires both REST and HAND_IMAGERY trials")
    rest_compliance = sum(row["success"] for row in rest) / len(rest)
    grasp_success = sum(row["success"] for row in imagery) / len(imagery)
    return {"episodes": len(results), "rest_compliance": rest_compliance,
            "grasp_success": grasp_success,
            "balanced_success": (rest_compliance + grasp_success) / 2,
            "unwanted_lifts_during_rest": sum(row["lifted"] for row in rest),
            "per_trial": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Existing decoder test_predictions.json; never use it to fit the policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/rl_policy/evaluation.json"))
    args = parser.parse_args()
    trials = load_predictions(args.predictions)
    env = GraspHoldEnv()
    try:
        model = PPO.load(args.model, device="cpu")
        reports = {
            "policy_decoded": measure(env, trials, model),
            "policy_oracle": measure(env, trials, model, use_oracle=True),
            "always_hold": measure(env, trials, "always_hold"),
            "always_grasp": measure(env, trials, "always_grasp"),
        }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    for name, report in reports.items():
        print(f"{name}: balanced={report['balanced_success']:.3f}, "
              f"grasp={report['grasp_success']:.3f}, rest={report['rest_compliance']:.3f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
