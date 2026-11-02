#!/usr/bin/env bash
set -Eeuo pipefail
REPO=/home/ubuntu/FRI-project-8b9db42
BUNDLE="$REPO/outputs/isaac/eeg-batch-transfer"
INPUTS="$REPO/outputs/isaac/eeg-batch-inputs"
install -m 0644 "$BUNDLE/commands.py" "$REPO/simulation/src/fri_bci_sim/commands.py"
install -m 0644 "$BUNDLE/executor.py" "$REPO/simulation/src/fri_bci_sim/executor.py"
if [[ -f "$BUNDLE/ppo_telemetry.py" ]]; then
  install -m 0644 "$BUNDLE/ppo_telemetry.py" "$REPO/simulation/src/fri_bci_sim/ppo_telemetry.py"
fi
install -m 0644 "$BUNDLE/controlled_lift.py" "$REPO/simulation/scripts/controlled_lift.py"
install -m 0755 "$BUNDLE/run_eeg_matched_batch.sh" "$REPO/simulation/cloud/run_eeg_matched_batch.sh"
mkdir -p "$INPUTS"
for n in 006 007 008 009 010; do
  install -m 0644 "$BUNDLE/S${n}R12-decisions.jsonl" "$INPUTS/S${n}R12-decisions.jsonl"
done
if [[ -f "$BUNDLE/source_manifest.json" ]]; then
  install -m 0644 "$BUNDLE/source_manifest.json" "$INPUTS/source_manifest.json"
fi
cd /home/ubuntu/IsaacLab-fri
.venv/bin/python -c 'from fri_bci_sim.commands import Skill; from fri_bci_sim.executor import SkillExecutor; print(Skill.GRASP_LIFT, SkillExecutor().state)'
.venv/bin/python "$REPO/simulation/scripts/controlled_lift.py" --help | grep -E -- '--decisions|--condition|always-hold|scripted-positive|scripted-pause'
parser_log=$(mktemp)
trap 'rm -f "$parser_log"' EXIT
if .venv/bin/python "$REPO/simulation/scripts/controlled_lift.py" \
    --output /tmp/fri-eeg-missing-trace-check.jsonl --condition temporal >"$parser_log" 2>&1; then
  echo "Missing EEG trace unexpectedly passed parser validation" >&2
  exit 1
fi
grep -q 'EEG command conditions require --decisions' "$parser_log" || {
  cat "$parser_log" >&2
  echo "Missing EEG trace did not fail closed as expected" >&2
  exit 1
}
bash -n "$REPO/simulation/cloud/run_eeg_matched_batch.sh"
printf 'deployed_source=%s\n' "$REPO/simulation/scripts/controlled_lift.py"
sha256sum "$REPO/simulation/scripts/controlled_lift.py" "$REPO/simulation/cloud/run_eeg_matched_batch.sh"
