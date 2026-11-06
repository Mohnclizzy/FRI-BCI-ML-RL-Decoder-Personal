"""Collect auditable physical outcomes and compact 120 Hz trace from five screens."""
import csv
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = root / 'outputs/isaac/unilateral-eval-5x12s-20261103T2310Z'
out = root / 'results/unilateral_2026-11-03'
out.mkdir(exist_ok=False)
manifest = json.loads((run / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
results = []
with (out / 'trace_120hz.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed','time_s','cube_x','cube_y','cube_z','left_contact_n','right_contact_n','hand_x','hand_y','hand_z'])
    for seed in manifest['seeds']:
        d = run / f'seed{seed}'
        summ = json.loads((d / 'ppo-telemetry.summary.json').read_text())
        rows = [json.loads(line) for line in (d / 'ppo-telemetry.jsonl').open()]
        start = [x for x in rows if x['record_type'] == 'episode_start']
        end = [x for x in rows if x['record_type'] in ('terminal_pre_reset','episode_incomplete_end')]
        samples = [x for x in rows if x['record_type'] == 'physics_step']
        assert len(start) == len(end) == 1 and len(summ['episode_results']) == 1
        ep = summ['episode_results'][0]
        assert len(samples) == ep['physics_samples'] and all(x['contact_valid'] for x in samples)
        assert ep['physics_step_gaps'] == ep['contact_missing_samples'] == 0
        assert sha(d / 'input/model.pt') == manifest['checkpoint_sha256']
        assert all(abs((b['episode_time_s']-a['episode_time_s'])-summ['physics_dt_s']) < 2e-6 for a,b in zip(samples,samples[1:]))
        for i, row in enumerate(samples):
            writer.writerow([seed, round(row['episode_time_s'], 6), *[round(v, 6) for v in (*row['cube_pose_w'][:3], *row['finger_contact_n'], *row['hand_pose_w'][:3])]])
        xy0 = start[0]['cube_pose_w'][:2]
        def first(predicate):
            row = next((x for x in samples if predicate(x)), None)
            return None if row is None else {'time_s':row['episode_time_s'], 'finger_positions_m':row['robot_joint_pos'][-2:], 'contact_n':row['finger_contact_n'], 'cube_pose_w':row['cube_pose_w']}
        events = {
            'first_right_only':first(lambda x:x['finger_contact_n'][1] >= .1 and x['finger_contact_n'][0] < .1),
            'first_xy_2cm_below_4cm':first(lambda x:math.dist(xy0,x['cube_pose_w'][:2]) >= .02 and x['cube_height_m'] < start[0]['baseline_cube_height_m'] + .04),
            'first_either_finger_at_most_20mm':first(lambda x:min(x['robot_joint_pos'][-2:]) <= .02),
            'first_both_fingers_at_most_20mm':first(lambda x:max(x['robot_joint_pos'][-2:]) <= .02),
            'first_rise_8cm':first(lambda x:x['cube_height_m'] >= start[0]['baseline_cube_height_m'] + .08),
        }
        maxxy = max(math.dist(xy0, x['cube_pose_w'][:2]) for x in samples)
        results.append({'seed':seed, 'events':events, **ep, 'start_cube_pose_w':start[0]['cube_pose_w'], 'end_record_type':end[0]['record_type'], 'end_cube_pose_w':end[0]['cube_pose_w'], 'max_cube_xy_displacement_m':maxxy, 'last_sample_lift_bilateral':samples[-1]['cube_height_m'] - start[0]['baseline_cube_height_m'] >= 0.08 and min(samples[-1]['finger_contact_n']) >= 0.1, 'terminal_half_second_lift_bilateral':len(samples) >= 60 and all(x['cube_height_m'] - start[0]['baseline_cube_height_m'] >= 0.08 and min(x['finger_contact_n']) >= 0.1 for x in samples[-60:]), 'last_sample_contacts_n':samples[-1]['finger_contact_n'], 'terminal_state':end[0], 'sampling_mode':summ['physics_sampling_mode'], 'physics_dt_s':summ['physics_dt_s'], 'policy_dt_s':summ['policy_dt_s'], 'exit':json.loads((d/'exit.json').read_text()), 'telemetry_sha256':sha(d/'ppo-telemetry.jsonl'), 'summary_sha256':sha(d/'ppo-telemetry.summary.json')})
(out / 'metrics.json').write_text(json.dumps({'manifest':manifest,'episodes':results,'trace_note':'Every recorded physics sample; fields rounded to 6 decimal places. Complete unrounded telemetry retained in ignored outputs.', 'trace_sha256':sha(out/'trace_120hz.csv')},indent=2)+'\n')
print(json.dumps(results,indent=2))
