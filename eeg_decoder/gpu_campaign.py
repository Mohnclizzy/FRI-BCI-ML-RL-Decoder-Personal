"""Resumable multi-seed EEG campaign. Target R12 is never loaded here.

Run plan before run. Work directory must be persistent; cache may be ephemeral.
One writer per shard. FP32 training keeps the original numerical convention.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import gpu_pipeline as p
from .alignment import apply_alignment
from .neural import EEGNet


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    base = p.read_config(plan['base_config'])
    for key in ('source_epochs', 'source_patience', 'source_batch_size', 'target_batch_size', 'checkpoint_every'):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(key)
    if not plan['target_epochs'] or sorted(set(plan['target_epochs'])) != plan['target_epochs'] or any(type(x) is not int or x < 1 for x in plan['target_epochs']):
        raise ValueError('target_epochs must be increasing positive integers')
    if not plan['seeds'] or len(set(plan['seeds'])) != len(plan['seeds']) or any(type(x) is not int or not 0 <= x < 2**31 for x in plan['seeds']):
        raise ValueError('Invalid seeds')
    if not plan['modes'] or not set(plan['modes']) <= {'scratch', 'transfer', 'head_only'} or len(set(plan['modes'])) != len(plan['modes']):
        raise ValueError('Invalid modes')
    if not plan['target_lrs'] or any(not np.isfinite(x) or x <= 0 for x in [plan['source_lr'], *plan['target_lrs']]):
        raise ValueError('Invalid learning rates')
    if len(set(plan['target_lrs'])) != len(plan['target_lrs']) or not plan['models']:
        raise ValueError('Duplicate learning rates or missing models')
    for name, args in plan['models'].items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Unsafe model name')
        EEGNet(**args)
    return plan, base


def shards(plan):
    return [(name, seed) for name in plan['models'] for seed in plan['seeds']]


def model_for(base, args):
    return EEGNet(channels=len(base['preprocessing']['channels']),
                  samples=round(base['preprocessing']['window_seconds'] * 160), **args)


def epoch(model, X, y, optimizer, seed, batch_size, head_only=False):
    p.seed_all(seed)
    model.train()
    if head_only:
        model.features.eval()  # Freeze batch-normalization state as well as weights.
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed)).to(y.device)
    losses = []
    for ids in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(X[ids]), y[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.constrain_weights()
        losses.append(loss.detach() * len(ids))
    return float(torch.stack(losses).sum().cpu() / len(y))


def tensors(X, y, device):
    return torch.as_tensor(X, dtype=torch.float32, device=device), torch.as_tensor(y, dtype=torch.long, device=device)


def checked_load(path, identity, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state['identity'] != identity:
        raise ValueError('Checkpoint provenance mismatch: ' + str(path))
    return state


def source_data(base, cache):
    train, labels, valid, truth, audit = [], [], [], [], []
    for subject in base['source_subjects']:
        fit = p.load_recording(subject, 4, base, cache)
        matrix = p.alignment_for(fit, base)
        for run in (4, 8, 12):
            rec = fit if run == 4 else p.load_recording(subject, run, base, cache)
            (valid if run == 12 else train).append(apply_alignment(rec['X'], matrix))
            (truth if run == 12 else labels).append(rec['y'])
            audit.append({'subject': subject, 'run': run, 'edf_sha256': rec['source_edf_sha256']})
        print(f'Prepared source S{subject:03d}', flush=True)
    return tuple(np.concatenate(v) for v in (train, labels, valid, truth)), audit


def train_source(plan, base, args, seed, folder, cache, identity, device):
    folder.mkdir(parents=True, exist_ok=True)
    data, audit = source_data(base, cache)
    identity = p.config_digest({'campaign': identity, 'source': audit})
    p.write_json(folder / 'data.json', audit)
    X, y = tensors(data[0], data[1], device)
    V, v = data[2:]
    p.seed_all(seed)
    model = model_for(base, args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=plan['source_lr'], weight_decay=1e-4)
    start, best, bad, history = 0, float('inf'), 0, []
    if (folder / 'last.pt').exists():
        s = checked_load(folder / 'last.pt', identity, device)
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        start, best, bad, history = s['epoch'], s['best'], s['bad'], s['history']
    for e in range(start, plan['source_epochs']):
        if bad >= plan['source_patience']:
            break
        started = time.monotonic()
        loss = epoch(model, X, y, opt, seed + e, plan['source_batch_size'])
        probs = p.predict(model, V, device)
        val = float(-np.log(probs[np.arange(len(v)), v].clip(1e-12)).mean())
        improved = val < best
        best, bad = (val, 0) if improved else (best, bad + 1)
        history.append({'epoch': e + 1, 'train_loss': loss, 'validation_loss': val, 'seconds': time.monotonic() - started})
        state = dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(), epoch=e+1, best=best, bad=bad, history=history)
        if improved:
            p.save_checkpoint(folder / 'best.pt', state)
        p.save_checkpoint(folder / 'last.pt', state)
        p.write_json(folder / 'history.json', history)
        print('source', history[-1], flush=True)
    checked_load(folder / 'best.pt', identity, 'cpu')
    return folder / 'best.pt'


def development(model, record, matrix, base, device):
    probs = p.predict(model, apply_alignment(record['X'], matrix), device)
    probabilities, times = p.replay_probabilities(model, record, matrix, base, device)
    commands = {}
    for name, threshold, persistence in [('raw', 0.50000001, 0), ('confidence', .7, 0), ('temporal', .7, .5)]:
        gate = p.HysteresisGate(threshold, 0, persistence, base['max_age_s'])
        skills = [gate.update(float(q), float(t), float(t), float(t), 'R08')['skill'] for q, t in zip(probabilities, times)]
        commands[name] = p.command_metrics(times, skills, record['intervals'], base['max_age_s'])
    return {'window_metrics': p.score(record['y'], probs.argmax(axis=1)), 'command_metrics': commands,
            'run': 8, 'role': 'development; not independent confirmation', 'target_r12_touched': False}


def train_target(plan, base, args, seed, mode, lr, subject, source, folder, cache, identity, device):
    folder.mkdir(parents=True, exist_ok=True)
    fit = p.load_recording(subject, 4, base, cache)
    dev = p.load_recording(subject, 8, base, cache)
    matrix = p.alignment_for(fit, base)
    source_sha = None if mode == 'scratch' else p.digest(source)
    identity = p.config_digest({'campaign': identity, 'mode': mode, 'lr': lr, 'subject': subject,
                                'fit': fit['source_edf_sha256'], 'dev': dev['source_edf_sha256'], 'source': source_sha})
    X, y = tensors(apply_alignment(fit['X'], matrix), fit['y'], device)
    p.seed_all(seed + subject)
    model = model_for(base, args).to(device)
    if mode != 'scratch':
        model.load_state_dict(torch.load(source, map_location=device, weights_only=True)['model'])
    if mode == 'head_only':
        model.features.requires_grad_(False)
    opt = torch.optim.Adam([x for x in model.parameters() if x.requires_grad], lr=lr, weight_decay=1e-4)
    start, history = 0, []
    if (folder / 'last.pt').exists():
        state = checked_load(folder / 'last.pt', identity, device)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        start, history = state['epoch'], state['history']
    # Re-score saved milestones after a disconnect between checkpoint and JSON writes.
    for e in plan['target_epochs']:
        checkpoint, result = folder / f'epoch_{e}.pt', folder / f'epoch_{e}.json'
        if checkpoint.exists() and not result.exists():
            saved = checked_load(checkpoint, identity, device)
            temporary = model_for(base, args).to(device)
            temporary.load_state_dict(saved['model'])
            p.write_json(result, {**development(temporary, dev, matrix, base, device), 'epoch': e,
                                 'model_sha256': p.digest(checkpoint), 'identity': identity})
    for e in range(start, max(plan['target_epochs'])):
        loss = epoch(model, X, y, opt, seed + 1000 * subject + e, plan['target_batch_size'], mode == 'head_only')
        history.append({'epoch': e + 1, 'loss': loss})
        state = dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(), epoch=e+1,
                     history=history, alignment=torch.as_tensor(matrix), source_sha256=source_sha)
        if e + 1 in plan['target_epochs']:
            checkpoint = folder / f'epoch_{e+1}.pt'
            p.save_checkpoint(checkpoint, state)
            metrics = development(model, dev, matrix, base, device)
            p.write_json(folder / f'epoch_{e+1}.json', {**metrics, 'epoch': e+1, 'identity': identity, 'model_sha256': p.digest(checkpoint)})
            print(f'{folder.name} {mode} lr={lr} epoch={e+1} BA={metrics["window_metrics"]["balanced_accuracy"]:.4f}', flush=True)
        if (e + 1) % plan['checkpoint_every'] == 0 or e + 1 in plan['target_epochs']:
            p.save_checkpoint(folder / 'last.pt', state)


def run(plan, base, work, cache, shard, device):
    name, seed = shards(plan)[shard]
    root = work / f'{name}_seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive-create lock survives crashes. Remove only after confirming old worker stopped.
    lock = root / 'RUNNING.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    try:
        code = {f.name: p.config_digest(f.read_text(encoding='utf-8')) for f in Path(__file__).parent.glob('*.py')}
        identity = p.config_digest({'plan': plan, 'base': base, 'code': code, 'shard': shard})
        manifest = root / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['identity'] != identity:
            raise ValueError('Campaign/code changed; use a new work directory')
        p.write_json(manifest, {**p.provenance(base), 'plan': plan, 'identity': identity, 'code': code,
                                'shard': shard, 'target_r12_touched': False})
        source = train_source(plan, base, plan['models'][name], seed, root / 'source', cache, identity, device)
        for mode in plan['modes']:
            for lr in plan['target_lrs']:
                for subject in base['target_subjects']:
                    target = root / mode / f'lr_{lr}' / f'S{subject:03d}'
                    train_target(plan, base, plan['models'][name], seed, mode, lr, subject, source, target, cache, identity, device)
        hashes = {str(f.relative_to(root)): p.digest(f) for f in root.rglob('*') if f.is_file() and f.name not in ('RUNNING.lock', 'complete.json')}
        p.write_json(root / 'complete.json', {'identity': identity, 'sha256': hashes})
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'summarize'])
    parser.add_argument('--config', default='configs/gpu_campaign.json')
    parser.add_argument('--work-dir', default='outputs/gpu_campaign')
    parser.add_argument('--cache-dir', default='outputs/gpu_campaign_cache')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--shard', type=int)
    a = parser.parse_args()
    plan, base = read_plan(a.config)
    jobs = shards(plan)
    if a.action == 'summarize':
        groups = {}
        for name in plan['models']:
            for mode in plan['modes']:
                for lr in plan['target_lrs']:
                    for e in plan['target_epochs']:
                        rows = []
                        for seed in plan['seeds']:
                            for subject in base['target_subjects']:
                                folder = Path(a.work_dir) / f'{name}_seed{seed}' / mode / f'lr_{lr}' / f'S{subject:03d}'
                                file = folder / f'epoch_{e}.json'
                                if not file.exists():
                                    continue
                                row = json.loads(file.read_text())
                                if p.digest(folder / f'epoch_{e}.pt') != row['model_sha256']:
                                    raise ValueError('Corrupted model: ' + str(file))
                                rows.append(row)
                        expected = len(plan['seeds']) * len(base['target_subjects'])
                        groups[f'{name}/{mode}/{lr}/{e}'] = {
                            'completed': len(rows), 'expected': expected, 'complete': len(rows) == expected,
                            'mean_balanced_accuracy': float(np.mean([r['window_metrics']['balanced_accuracy'] for r in rows])) if rows else None,
                            'mean_raw_rest_activation': float(np.mean([r['command_metrics']['raw']['rest_command_fraction'] for r in rows])) if rows else None,
                            'mean_raw_missed_imagery': float(np.mean([r['command_metrics']['raw']['missed_imagery_fraction'] for r in rows])) if rows else None}
        destination = Path(a.work_dir) / 'development_summary.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        p.write_json(destination, {'groups': groups, 'claim': 'Exploratory R08 development. Seeds are not independent subjects. Do not rank incomplete groups.'})
        print(destination)
        return
    if a.action == 'plan':
        count = len(jobs) * len(plan['modes']) * len(plan['target_lrs']) * len(base['target_subjects'])
        print(json.dumps({'shards': list(enumerate(jobs)), 'source_runs': len(jobs), 'target_trajectories': count,
                          'target_scored_checkpoints': count * len(plan['target_epochs']), 'plan_sha256': p.config_digest(plan),
                          'target_r12_access': False}, indent=2))
        return
    if a.shard is not None and not 0 <= a.shard < len(jobs):
        parser.error('Invalid shard')
    device = p.device_for(a.device)
    torch.set_num_threads(4)
    for index in range(len(jobs)) if a.shard is None else [a.shard]:
        run(plan, base, Path(a.work_dir), Path(a.cache_dir) / f'shard{index}', index, device)


if __name__ == '__main__':
    main()
