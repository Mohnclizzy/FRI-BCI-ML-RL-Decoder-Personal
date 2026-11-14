"""Frozen R04->R08 command comparison; never opens target R12 or retrains EEGNet.

python -m eeg_decoder.bounded_study --campaign-dir PATH --output NEW_DIRECTORY
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import gpu_pipeline as p
from .gpu_campaign import model_for, read_plan
from .alignment import apply_alignment
from .risk_gate import HysteresisGate, calibrate_threshold, rest_episode_maxima
from . import command_study_metrics as m

SUBJECTS = list(range(1, 16))
SEEDS = [42, 123, 2026]


def gate_specs(probs, times, intervals, base):
    stats = m.analyze_transition_vs_interior(probs, times, intervals)
    q = stats['pure_rest_interior']['q80']
    q = float(np.clip(q or 0.55, 0.50, 0.95))
    crc = calibrate_threshold(rest_episode_maxima(probs, times, intervals, base['max_age_s']), base['alpha'])
    return {
        'raw': dict(threshold=float(np.nextafter(0.5, 1)), margin=0., persistence_s=0.),
        'confidence': dict(threshold=0.70, margin=0., persistence_s=0.),
        'temporal': dict(threshold=0.70, margin=0., persistence_s=0.50),
        'moderate': dict(threshold=0.55, margin=0.02, persistence_s=0.25),
        'rest_q80': dict(threshold=q, margin=0.02, persistence_s=0.25),
        'crc': dict(threshold=crc['threshold'], margin=base['hysteresis_margin'], persistence_s=base['persistence_s']),
    }


def utility(row):
    return (row['enabled_seconds_20s'] >= 2.5 and row['imagery_command_coverage'] >= .30
            and row['rest_command_fraction'] <= .35)


def aggregate(rows):
    """Average seed metrics within a person, then score that person's utility."""
    participants = []
    for name in sorted({r['name'] for r in rows}):
        for subject in SUBJECTS:
            group = [r for r in rows if r['name'] == name and r['subject'] == subject]
            if not group:
                raise ValueError(f'Missing participant: {name} S{subject:03d}')
            expected = SEEDS if name.startswith('eegnet/') else [None]
            if sorted(str(r['seed']) for r in group) != sorted(map(str, expected)):
                raise ValueError('Incomplete or duplicate seeds: ' + name)
            metrics = {}
            for key in group[0]['metrics']:
                vals = [r['metrics'][key] for r in group if r['metrics'][key] is not None]
                metrics[key] = float(np.mean(vals)) if vals else None
            metrics['passes_predeclared_utility'] = utility(metrics)
            participants.append(dict(name=name, subject=subject, metrics=metrics))
    groups = []
    for name in sorted({r['name'] for r in participants}):
        group = [r['metrics'] for r in participants if r['name'] == name]
        groups.append(dict(name=name, participants=len(group),
            pass_fraction=float(np.mean([r['passes_predeclared_utility'] for r in group])),
            rest_command_fraction=float(np.mean([r['rest_command_fraction'] for r in group])),
            imagery_command_coverage=float(np.mean([r['imagery_command_coverage'] for r in group]))))
    groups.sort(key=lambda r: (-r['pass_fraction'], r['rest_command_fraction'], -r['imagery_command_coverage'], r['name']))
    return participants, groups


def verify_checkpoint(root, seed, subject, fit, dev, base, plan):
    shard = root / f'eegnet8_seed{seed}'
    manifest = json.loads((shard / 'manifest.json').read_text())
    complete = json.loads((shard / 'complete.json').read_text())
    if manifest['plan'] != plan or manifest['config_sha256'] != p.config_digest(base):
        raise ValueError('Saved campaign configuration differs')
    if manifest['identity'] != complete['identity']:
        raise ValueError('Completion identity differs')
    def checked(rel):
        path = shard / rel
        if p.digest(path) != complete['sha256'][rel]:
            raise ValueError('Artifact hash mismatch: ' + str(path))
        return path
    checked('manifest.json')
    source_path = checked('source/best.pt')
    if p.digest(source_path) != complete['sha256']['source/best.pt']:
        raise ValueError('Source checkpoint hash mismatch')
    rel = f'transfer/lr_0.0001/S{subject:03d}/epoch_300'
    checkpoint, result = checked(rel + '.pt'), checked(rel + '.json')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    result = json.loads(result.read_text())
    expected = p.config_digest(dict(campaign=manifest['identity'], mode='transfer', lr=.0001, subject=subject,
        fit=fit['source_edf_sha256'], dev=dev['source_edf_sha256'], source=state['source_sha256']))
    if (state['epoch'] != 300 or state['identity'] != expected or result['identity'] != expected
            or result['model_sha256'] != p.digest(checkpoint)
            or state['source_sha256'] != complete['sha256']['source/best.pt']):
        raise ValueError('Checkpoint provenance mismatch: ' + str(checkpoint))
    matrix = state['alignment'].numpy()
    if not np.allclose(matrix, p.alignment_for(fit, base), rtol=1e-5, atol=1e-5):
        raise ValueError('Alignment differs from R04 fit')
    model = model_for(base, plan['models']['eegnet8'])
    model.load_state_dict(state['model'])
    return model, matrix, dict(path=str(checkpoint), sha256=p.digest(checkpoint), identity=expected,
                              saved_balanced_accuracy=result['window_metrics']['balanced_accuracy'])


def trace_for(probs, times, spec, subject, max_age):
    gate = HysteresisGate(**spec, max_age_s=max_age)
    return [dict(estimate=dict(source=f'S{subject:03d}R08', episode='R08', window_end_s=float(t),
                    available_at_s=float(t), probability=float(prob)),
                 decisions={'raw': gate.update(float(prob), float(t), float(t), float(t), 'R08')})
            for prob, t in zip(probs, times)]


def schedule(trace, max_age=1.):
    times = np.array([r['estimate']['available_at_s'] for r in trace])
    steps = []
    for step in range(400):
        t = step / 20
        i = int(np.searchsorted(times, t, side='right')) - 1
        if i >= 0 and t - times[i] <= max_age and trace[i]['decisions']['raw']['skill'] == 'grasp_lift':
            steps.append(step + 1)
    return steps


def spans(steps):
    result = []
    for step in sorted(set(steps)):
        if result and result[-1][1] == step:
            result[-1][1] += 1
        else:
            result.append([step, step + 1])
    return result


def matched_schedules(steps):
    return {'contiguous': spans(range(1, len(steps) + 1)),
            'shifted': spans(((s - 1 + 200) % 400) + 1 for s in steps)}


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    plan, base = read_plan('configs/gpu_campaign.json')
    if base['target_subjects'] != SUBJECTS or plan['seeds'] != SEEDS:
        raise ValueError('Frozen cohort/seeds changed')
    protocol = dict(version=1, subjects=SUBJECTS, training_seeds=SEEDS, robot_seeds=[0, 1, 2],
        model='eegnet8/transfer/0.0001/300', target_fit_run=4, development_run=8, target_r12_touched=False,
        selection='maximize participant pass fraction after seed averaging; minimize REST; maximize imagery; name',
        utility=dict(enabled_seconds_20s=2.5, imagery_coverage=.3, max_rest_fraction=.35),
        horizon_s=20, control_hz=20, guarantee_claimed=False,
        claim='Exploratory R08 development; thresholds selected/scored on same recording; no independent confirmation',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        python=platform.python_version(), config=base, campaign_plan=plan)
    p.write_json(out / 'protocol.json', protocol)
    (out / 'scores').mkdir(); (out / 'traces').mkdir(); (out / 'schedules').mkdir()
    rows, evidence = [], []
    torch.set_num_threads(4)
    for subject in SUBJECTS:
        fit = p.load_recording(subject, 4, base, Path(args.cache_dir))
        dev = p.load_recording(subject, 8, base, Path(args.cache_dir))
        windows, times = m.extract_continuous_4hz_windows(dev['signal'])
        entries = []
        for name, constructor in [('csp', m.make_csp_lda), ('riemannian', m.make_riemannian_lda)]:
            model = constructor().fit(fit['X'], fit['y'])
            entries.append((name, None, model.predict_proba(dev['X'])[:, 1], model.predict_proba(windows)[:, 1]))
        for seed in SEEDS:
            model, matrix, artifact = verify_checkpoint(Path(args.campaign_dir), seed, subject, fit, dev, base, plan)
            cue = p.predict(model, apply_alignment(dev['X'], matrix), 'cpu')[:, 1]
            probs = p.predict(model, apply_alignment(windows, matrix), 'cpu')[:, 1]
            # Different CPU/GPU arithmetic may flip exact-boundary predictions. Record, never silently replace.
            artifact.update(subject=subject, seed=seed, reproduced_balanced_accuracy=m.score_cue_windows(dev['y'], cue)['balanced_accuracy'])
            evidence.append(artifact)
            entries.append(('eegnet', seed, cue, probs))
        p.write_json(out / 'scores' / f'S{subject:03d}_annotations.json', dev['intervals'])
        for decoder, seed, cue, probs in entries:
            prefix = f'{decoder}_seed{seed}_S{subject:03d}'
            np.savez_compressed(out / 'scores' / f'{prefix}.npz', times=times, probabilities=probs, cue_probabilities=cue, cue_labels=dev['y'])
            classification = m.score_cue_windows(dev['y'], cue)
            for gate, spec in gate_specs(probs, times, dev['intervals'], base).items():
                metrics = m.evaluate_gate_on_r08(probs, times, dev['intervals'], **spec, max_age_s=base['max_age_s'])
                trace = trace_for(probs, times, spec, subject, base['max_age_s'])
                path = out / 'traces' / f'{prefix}_{gate}.jsonl'
                path.write_text(''.join(json.dumps(r) + '\n' for r in trace), encoding='utf-8')
                if len(schedule(trace, base['max_age_s'])) != metrics['enabled_steps_20s']:
                    raise ValueError('Robot clock differs from command metric')
                rows.append(dict(name=f'{decoder}/{gate}', subject=subject, seed=seed, metrics=metrics,
                                 classification=classification, trace=str(path.relative_to(out))))
        p.write_json(out / 'rows.json', rows)
        print(f'Completed S{subject:03d}: {len(rows)} decoder/seed/gate rows', flush=True)
    participants, groups = aggregate(rows)
    selected = groups[0] if groups[0]['pass_fraction'] > 0 else None
    p.write_json(out / 'artifact_verification.json', evidence)
    p.write_json(out / 'participants.json', participants)
    p.write_json(out / 'groups.json', groups)
    selection = dict(selected=selected, protocol_sha256=p.digest(out/'protocol.json'), groups_sha256=p.digest(out/'groups.json'))
    p.write_json(out / 'selection.json', selection)
    jobs = []
    if selected:
        decoder, gate = selected['name'].split('/')
        seed = 42 if decoder == 'eegnet' else None
        for subject in SUBJECTS:
            by_gate = {r['name'].split('/')[1]: r for r in rows if r['subject'] == subject and r['seed'] == seed and r['name'].startswith(decoder+'/')}
            path = out / by_gate[gate]['trace']
            active = schedule([json.loads(line) for line in path.read_text().splitlines()], base['max_age_s'])
            controls = matched_schedules(active)
            for key, control in controls.items():
                p.write_json(out/'schedules'/f'S{subject:03d}_{key}.json', dict(spans=control))
            for robot_seed in (0, 1, 2):
                for condition in ('selected', 'raw', 'always-enable', 'always-hold', 'contiguous', 'shifted'):
                    job = dict(subject=subject, seed=robot_seed, condition=condition)
                    if condition in ('selected', 'raw'):
                        job.update(harness_condition='eeg', gate='raw', trace=by_gate[gate if condition=='selected' else 'raw']['trace'])
                    elif condition in controls:
                        job.update(harness_condition='enable-schedule', schedule=f'schedules/S{subject:03d}_{condition}.json')
                    else:
                        job.update(harness_condition=condition)
                    jobs.append(job)
    p.write_json(out/'robot_jobs.json', dict(selection_sha256=p.digest(out/'selection.json'), jobs=jobs,
        representative_video=dict(subject=1, seed=0, condition='selected')))
    plot(out, groups)
    files = {str(f.relative_to(out)): p.digest(f) for f in out.rglob('*') if f.is_file()}
    p.write_json(out/'SHA256SUMS.json', files)
    print(json.dumps(selection, indent=2), flush=True)


def plot(out, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    for row in groups:
        ax.scatter(row['rest_command_fraction'], row['imagery_command_coverage'], s=40+250*row['pass_fraction'])
        ax.annotate(row['name'], (row['rest_command_fraction'], row['imagery_command_coverage']), fontsize=8, xytext=(4,4), textcoords='offset points')
    ax.axvline(.35, color='gray', ls='--'); ax.axhline(.30, color='gray', ls='--')
    ax.set(xlabel='REST command fraction (lower is better)', ylabel='Imagery command coverage (higher is better)',
           title='R08 development command tradeoff — 15 participants; seeds averaged within participant', xlim=(-.03,1), ylim=(-.03,1))
    fig.tight_layout(); fig.savefig(out/'command_tradeoff.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', required=True)
    parser.add_argument('--cache-dir', default='outputs/local_scratch_2026-11-11')
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
