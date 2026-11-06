#!/usr/bin/env bash
# S001R12 PHASE-B MATCHED CONTROLS — unambiguous duty-cycle bounds, latch z-offset +0.015 m.
# always-hold  (0% command duty)  x seeds 0-2
# always-enable (100% command duty) x seeds 0-2
# NOTE: the `pause`-matched control is HELD BACK here — it is inverted in the harness
# (results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py lines 85-86: hold INSIDE the
# block) relative to the frozen spec (freeze doc S7: policy ENABLED inside the contiguous block,
# disabled elsewhere). See docs/CONNECTED_BOTTLENECK_STAGE1_ADDENDUM_2026-11-06.md S1.
# Identical harness/regime/checkpoint as the S001R12 EEG arm (raw/confidence/temporal x seeds 0-2).
set -uo pipefail
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal
export MUJOCO_PY_MUJOCO_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210
export LD_LIBRARY_PATH=/home/ubuntu/robomimic-native-phase2-20261104/runtime/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/ubuntu/robomimic-native-phase2-20261104/src/robosuite:./simulation/src
PYTHON=/home/ubuntu/robomimic-native-phase2-20261104/.venv/bin/python
SCRIPT=results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py
OFF=0.015
BASE=results/robomimic_controls_s001r12_2026-11-06
mkdir -p "$BASE"
for cond in always-hold always-enable; do
  for seed in 0 1 2; do
    name=${cond}_seed${seed}
    OUT=$BASE/$name
    if [ -d "$OUT" ]; then echo "[skip] $name (dir exists)"; continue; fi
    "$PYTHON" "$SCRIPT" --seed "$seed" --condition "$cond" --latch-z-offset "$OFF" --out "$OUT" > "$BASE/$name.log" 2>&1
    echo "[$(date +%H:%M:%S)] $name exit=$?"
  done
done
# --- aggregate ---
cat > "$BASE/aggregate.py" <<'PY'
import json, pathlib
base = pathlib.Path('results/robomimic_controls_s001r12_2026-11-06')
def label(s):
    if s.get('terminal_retained'): return 'complete_retained'
    if s.get('first_acquisition_step') is None: return 'contact_none'
    if s.get('post_acquisition_loss'): return 'contact_lost'
    return 'complete_unretained'
out = []
for d in sorted(p for p in base.iterdir() if (p/'summary.json').exists()):
    s = json.loads((d/'summary.json').read_text())
    out.append({'condition': s['condition'], 'seed': s['seed'],
                'label': label(s), 'terminal_retained': bool(s.get('terminal_retained')),
                'first_acquisition_step': s.get('first_acquisition_step'),
                'max_rise_m': s.get('max_rise_m'), 'checkpoint_sha256': s.get('checkpoint_sha256')})
out.sort(key=lambda r: (r['condition'], r['seed']))
with (base/'controls.tsv').open('w') as f:
    f.write('condition\tseed\tcompletion_state\tterminal_retained\tfirst_acquisition_step\tmax_rise_m\n')
    for r in out:
        f.write('\t'.join(str(r[k]) for k in ['condition','seed','label','terminal_retained','first_acquisition_step','max_rise_m'])+'\n')
print('\n'.join('\t'.join(str(r[k]) for k in ['condition','seed','label','terminal_retained','first_acquisition_step','max_rise_m']) for r in out))
print('checkpoint', out[0]['checkpoint_sha256'] if out else None)
PY
"$PYTHON" "$BASE/aggregate.py" 2>&1 || python3 "$BASE/aggregate.py"
echo "CONTROLS_S001_BATCH_COMPLETE runs=$(ls "$BASE" | grep -v '\\.log$' | grep -vE '\\.(sh|py|tsv)$' | grep -cv '^aggregate')"
