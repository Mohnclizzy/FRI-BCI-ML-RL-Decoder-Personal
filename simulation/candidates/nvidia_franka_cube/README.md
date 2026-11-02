# NVIDIA pretrained Franka Cube candidate

This is an additive, opt-in compatibility port of NVIDIA's downloaded PPO policy.
It does not replace or register the project's FRI tasks, install a package into the
Isaac runtime, change the pinned runtime, train weights, or connect EEG.

The official archive and checkpoint hashes are fixed in `prepare.py`. Original
download bytes are preserved; every preparation and playback destination must be
new. Checkpoints and third-party source remain under ignored outputs, not in Git.

## Reproduce on Brev

Run from the Personal project checkout. Choose new paths every time:

```bash
PY=/home/ubuntu/IsaacLab-fri/.venv/bin/python
CANDIDATE=simulation/candidates/nvidia_franka_cube
BASE=outputs/isaac/nvidia-candidate-NEW-RUN-ID

$PY "$CANDIDATE/prepare.py" \
  --archive outputs/policy_candidates/nvidia_franka_cube.zip --work "$BASE/prepared"

# Checkpoint tensor dimensions, normalization schema, config imports and runtime pin.
# This mode does not construct an environment or launch simulation.
$PY "$CANDIDATE/run.py" --probe --viz none \
  --candidate-root "$BASE/prepared/franka_cube" \
  --isaaclab-root /home/ubuntu/IsaacLab-fri --output "$BASE/probe"

# First startup gate: one native-horizon episode; inspect summary and trace.
$PY "$CANDIDATE/run.py" --episodes 1 --seed 8 --viz none \
  --candidate-root "$BASE/prepared/franka_cube" \
  --isaaclab-root /home/ubuntu/IsaacLab-fri --output "$BASE/startup"

# Until the in-process Newton reset mismatch is resolved, use five fresh
# simulator processes. Each one-episode output is exclusive and checked offline.
for seed in 9 10 11 12 13; do
  $PY "$CANDIDATE/run.py" --episodes 1 --seed "$seed" --viz none \
    --candidate-root "$BASE/prepared/franka_cube" \
    --isaaclab-root /home/ubuntu/IsaacLab-fri --output "$BASE/seed-$seed"
  python "$CANDIDATE/verify_saved.py" "$BASE/seed-$seed"
done
```

An unsuccessful physical outcome is retained. Do not replace episodes or expand
training. Each invocation is finite; there is no optimization call. Native timeout
logic ends episodes at about 2.9833 s (179 steps at 60 Hz), despite a configured
3 s horizon. Preserve this fact when reporting the native condition.
`--episodes 5` remains a diagnostic mode, but rejects the run if any first
physics sample disagrees with its reset state. The original five-episode run
failed that check for episodes 2–5; see the reset diagnostic report.

## Compatibility and measurement

- Target runtime: Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.
- Select the legacy Franka asset configuration expected by the original source;
  map deprecated velocity/effort-limit assignments to current names. Static
  joint friction stays static joint friction. Do not merge shoulder/forearm gains.
- The current importer requires cfg-owned assets/clone plans. The harness declares
  the original robot, cube, table, ground and light configurations/transforms to
  the scene, then reuses the original reset, rewards, observations and action logic.
  It bypasses the obsolete manual `_setup_scene` and clone call.
- Use Isaac Lab's RSL-RL compatibility conversion; load actor and normalization
  through the installed runner, enter evaluation mode and dispatch deterministic
  inference. Keep all eight actions and 41 observations.
- Add explicit finger-to-cube contact sensors. Read actual force matrices rather
  than the native grasp-quality proxy. Sensor reporting is an instrumentation
  addition and is not itself a guarantee of physically equivalent execution.
- Observe existing scene updates without changing graph/decimation/actuator flags.
  Log actual sampling intervals. First state precedes policy action; terminal
  state precedes autoreset. Unfinished/error cases are recorded and fail the gate.
- Preserve and count original gripper-enclosure assistance and joint-state limit
  corrections. This is a learned policy with engineered assistance, not an
  unconstrained learned controller. Counts are cumulative, with each episode's
  start/end values saved; maxima are cumulative as well.
- Sampled lift: >=0.08 m rise and each finger force >=0.1 N over a timestamp span
  >=0.5 s. Drop: after acquisition, below baseline +0.04 m over >=0.25 s. Sampling
  cannot prove uninterrupted contact between samples. Missing contact data makes
  the success measurement unavailable and prevents a completed validation gate.

This port is a separate condition, not an exact native-runtime reproduction or
a controlled comparison to the prior 12-second project evaluation. A successful
short playback does not establish retention, robustness, or HOLD/resume behavior.
Those require separately specified tests before EEG integration.

## Tests

```bash
python -m unittest discover -s simulation/candidates/nvidia_franka_cube -p test_candidate.py -v
```

Tests cover elapsed-time scoring, unilateral/missing contact, telemetry gaps,
drop dwell, timestamp ordering, archive integrity and preservation/path safety.
See `docs/NVIDIA_CANDIDATE_INTEGRATION_2026-11-02.md` for the measured run record.
