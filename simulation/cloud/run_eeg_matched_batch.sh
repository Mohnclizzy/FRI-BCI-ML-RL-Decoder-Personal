#!/usr/bin/env bash
# One bounded matched run: five existing EEG recordings x three gates, plus controls.
# Frozen corrected exploratory batch: 120 s matched horizon, same fixed scene.
# Set CONTROLS_ONLY=1 to run the 24/24/34 s control validation first.
set -Eeuo pipefail

: "${FRI_REPO:=/home/ubuntu/FRI-project-8b9db42}"
: "${ISAACLAB_ROOT:=/home/ubuntu/IsaacLab-fri}"
: "${BATCH_STAMP:=$(date -u +%Y%m%dT%H%M%SZ)}"
: "${BUDGET_S:=120}"
: "${CONTROLS_ONLY:=0}"
INPUT_DIR="$FRI_REPO/outputs/isaac/eeg-batch-inputs"
RUN_DIR="$FRI_REPO/outputs/isaac/eeg-matched-5p-$BATCH_STAMP-b${BUDGET_S}s"
PY="$ISAACLAB_ROOT/.venv/bin/python"
SCRIPT="$FRI_REPO/simulation/scripts/controlled_lift.py"

[[ -x "$PY" ]] || { echo "Missing pinned Isaac Lab Python: $PY" >&2; exit 2; }
[[ -f "$SCRIPT" ]] || { echo "Missing executor script: $SCRIPT" >&2; exit 2; }
[[ "$BUDGET_S" =~ ^[1-9][0-9]*$ ]] || { echo "BUDGET_S must be a positive integer" >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite $RUN_DIR" >&2; exit 2; }
if [[ "$CONTROLS_ONLY" == "0" ]]; then
  for n in 006 007 008 009 010; do
    [[ -s "$INPUT_DIR/S${n}R12-decisions.jsonl" ]] || { echo "Missing decisions for S$n" >&2; exit 2; }
  done
elif [[ "$CONTROLS_ONLY" != "1" ]]; then
  echo "CONTROLS_ONLY must be 0 or 1" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
cd "$ISAACLAB_ROOT"
printf 'condition\texit_code\ttelemetry\n' > "$RUN_DIR/results.tsv"
printf 'stamp=%s\nbudget_s=%s\ncontrols_only=%s\nexploratory_post_hoc=true\nrepo=%s\nisaaclab=%s\n' \
  "$BATCH_STAMP" "$BUDGET_S" "$CONTROLS_ONLY" "$FRI_REPO" "$ISAACLAB_ROOT" > "$RUN_DIR/manifest.txt"
if [[ -s "$INPUT_DIR/source_manifest.json" ]]; then
  cp -- "$INPUT_DIR/source_manifest.json" "$RUN_DIR/source_manifest.json"
fi
sha256sum "$SCRIPT" "$FRI_REPO/simulation/src/fri_bci_sim/commands.py" \
  "$FRI_REPO/simulation/src/fri_bci_sim/executor.py" \
  "$FRI_REPO/simulation/src/fri_bci_sim/tasks/lift.py" \
  "$(realpath "${BASH_SOURCE[0]}")" >> "$RUN_DIR/manifest.txt"
if [[ "$CONTROLS_ONLY" == "0" ]]; then
  sha256sum "$INPUT_DIR"/S*R12-decisions.jsonl >> "$RUN_DIR/manifest.txt"
fi

run_case() {
  local name="$1"
  local seconds="$2"
  shift 2
  local case_dir="$RUN_DIR/$name"
  mkdir -p "$case_dir"
  echo "START $name $(date -u +%FT%TZ)"
  set +e
  timeout 360s "$PY" "$SCRIPT" --device cuda --joint-waypoints --seconds "$seconds" \
    --output "$case_dir/telemetry.jsonl" "$@" > "$case_dir/run.log" 2>&1
  local status=$?
  set -e
  printf '%s\t%s\t%s\n' "$name" "$status" "$case_dir/telemetry.jsonl" >> "$RUN_DIR/results.tsv"
  tail -n 3 "$case_dir/run.log" || true
  echo "END $name exit=$status $(date -u +%FT%TZ)"
  if [[ $status -ne 0 || ! -s "$case_dir/telemetry.jsonl" ]]; then
    echo "Batch stopped: invalid/failed case $name; inspect $case_dir/run.log" >&2
    exit 1
  fi
  "$PY" - "$case_dir/run.log" "$seconds" <<'PY'
import json
import pathlib
import sys

log = pathlib.Path(sys.argv[1])
requested = float(sys.argv[2])
summary = None
for line in reversed(log.read_text(errors="replace").splitlines()):
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "simulated_duration_s" in value:
        summary = value
        break
if summary is None:
    raise SystemExit(f"No simulation-duration summary found in {log}")
dt = float(summary["physics_dt_s"])
actual = float(summary["simulated_duration_s"])
steps = int(summary["physics_steps"])
if steps <= 0 or abs(actual - requested) > dt + 1e-6:
    raise SystemExit(f"Run {log.parent.name} ended at {actual}s ({steps} steps), requested {requested}s ± {dt}s")
print(f"duration_check requested={requested} simulated={actual} dt={dt} steps={steps}")
PY
}

check_control() {
  local name="$1"
  local expected="$2"
  local log="$RUN_DIR/$name/run.log"
  "$PY" - "$log" "$expected" <<'PY'
import json
import pathlib
import sys

log = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
summary = None
for line in reversed(log.read_text(errors="replace").splitlines()):
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "physical_lift_pass" in value:
        summary = value
        break
if summary is None:
    raise SystemExit(f"No physical-lift summary found in {log}")
print(json.dumps(summary, sort_keys=True))
if expected == "hold":
    checks = {
        "physical_lift_pass": summary["physical_lift_pass"] is False,
        "max_hand_displacement_m": summary["max_hand_displacement_m"] <= 0.02,
        "max_joint_displacement_rad": summary["max_joint_displacement_rad"] <= 0.1,
        "max_hold_hand_displacement_m": summary["max_hold_hand_displacement_m"] <= 0.02,
        "max_hold_joint_displacement_rad": summary["max_hold_joint_displacement_rad"] <= 0.1,
        "max_hold_target_change_rad": summary["max_hold_target_change_rad"] <= 1e-6,
        "max_hold_finger_target_change_m": summary["max_hold_finger_target_change_m"] <= 1e-6,
    }
else:
    checks = {
        "physical_lift_pass": summary["physical_lift_pass"] is True,
        "dropped_after_acquisition": summary["dropped_after_acquisition"] is False,
        "skill_complete": summary["skill_complete"] is True,
        "max_hold_target_change_rad": summary["max_hold_target_change_rad"] <= 1e-6,
        "max_hold_finger_target_change_m": summary["max_hold_finger_target_change_m"] <= 1e-6,
    }
if not all(checks.values()):
    raise SystemExit(f"Control {log.parent.name} failed: {checks}")
PY
}

run_case always-hold 24 --condition always-hold
check_control always-hold hold
run_case scripted-positive 24 --condition scripted-positive
check_control scripted-positive positive
run_case scripted-pause 34 --condition scripted-pause
check_control scripted-pause positive
if [[ "$CONTROLS_ONLY" == "1" ]]; then
  echo "CONTROLS_COMPLETE $RUN_DIR"
  exit 0
fi
[[ "$CONTROLS_ONLY" == "0" ]] || { echo "CONTROLS_ONLY must be 0 or 1" >&2; exit 2; }
for n in 006 007 008 009 010; do
  for condition in raw confidence temporal; do
    run_case "S${n}R12-$condition" "$BUDGET_S" \
      --decisions "$INPUT_DIR/S${n}R12-decisions.jsonl" --condition "$condition"
  done
done

echo "BATCH_COMPLETE $RUN_DIR"
