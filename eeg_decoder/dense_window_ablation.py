"""Matched dense-causal-window ablation for the frozen EEGNet8 transfer recipe.

Runs the same number of optimizer updates as the selected cue-window recipe.
Run: python -m eeg_decoder.dense_window_ablation --config configs/gpu_research.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from . import gpu_pipeline as p
from .gpu_campaign import development
from .alignment import apply_alignment
from .neural import EEGNet


SEEDS = (42, 123, 2026)
EPOCHS = 300
SOURCE_BATCH = 128
TARGET_BATCH = 15


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dense_examples(record, sfreq=160, window_s=2., stride_s=.25):
    """Causal [end-window,end] windows labeled by instructed state at end."""
    signal = record['signal']
    width, step = round(sfreq * window_s), round(sfreq * stride_s)
    ends = np.arange(width, signal.shape[1] + 1, step, dtype=np.int64)
    intervals = record['intervals']
    starts = np.asarray([v[0] for v in intervals], dtype=float)
    stops = np.asarray([v[1] for v in intervals], dtype=float)
    labels = np.asarray([v[2] for v in intervals], dtype=np.int64)
    ts = ends / float(sfreq)
    ids = np.searchsorted(stops, ts, side='right')
    ok = (ids < len(stops))
    if not ok.all():
        ends, ts, ids = ends[ok], ts[ok], ids[ok]
    valid = (ts >= starts[ids] - 1e-9) & (ts <= stops[ids] + 1e-9)
    ends, ts, ids = ends[valid], ts[valid], ids[valid]
    if len(ends) == 0:
        raise ValueError(f"No dense windows for S{record['subject']:03d}R{record['run']:02d}")
    windows = np.stack([signal[:, e-width:e] for e in ends]).astype(np.float32)
    ys = labels[ids]
    episode_ids = np.asarray([record['subject'] * 100000 + record['run'] * 1000 + int(i) for i in ids])
    return windows, ys, episode_ids, ts


def episode_sampler(labels, episode_ids, n_draws, batch_size, seed):
    """Yield deterministic participant/episode-balanced batches with equal states."""
    rng = np.random.default_rng(seed)
    by_class_episode = []
    for cls in (0, 1):
        mapping = {}
        for idx in np.flatnonzero(labels == cls):
            mapping.setdefault(int(episode_ids[idx]), []).append(int(idx))
        choices = [(k, np.asarray(v, dtype=np.int64)) for k, v in sorted(mapping.items())]
        if not choices:
            raise ValueError(f"Dense training partition lacks class {cls}")
        by_class_episode.append(choices)
    half = batch_size // 2
    steps = math.ceil(n_draws / batch_size)
    for _ in range(steps):
        ids = []
        for cls in (0, 1):
            episodes = by_class_episode[cls]
            for _ in range(half):
                _, indices = episodes[int(rng.integers(len(episodes)))]
                ids.append(int(indices[int(rng.integers(len(indices)))]))
        yield np.asarray(ids, dtype=np.int64)


def model_for(c):
    return EEGNet(channels=len(c['preprocessing']['channels']),
                  samples=round(c['preprocessing']['window_seconds'] * 160),
                  f1=8, depth=2, f2=16, dropout=.25)


def update(model, X, y, optimizer, indices, device):
    model.train(); optimizer.zero_grad(set_to_none=True)
    xb = torch.as_tensor(X[indices], dtype=torch.float32, device=device)
    yb = torch.as_tensor(y[indices], dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    if not torch.isfinite(loss): raise FloatingPointError('non-finite dense-window training loss')
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step(); model.constrain_weights()
    return float(loss.detach().cpu())


def cue_updates(X, batch):
    return math.ceil(len(X) / batch)


def train_partition(model, optimizer, X, y, episodes, update_count, batch, seed, device, label):
    history = []
    tick = time.monotonic()
    for step, ids in enumerate(episode_sampler(y, episodes, update_count * batch, batch, seed), 1):
        loss = update(model, X, y, optimizer, ids, device)
        if step % max(1, update_count // 20) == 0 or step == update_count:
            record = dict(partition=label, update=step, update_budget=update_count, loss=loss,
                          elapsed_s=time.monotonic() - tick, gpu_mem_allocated_bytes=torch.cuda.max_memory_allocated())
            print(json.dumps(record), flush=True); history.append(record)
    return history


def fit_source(base, cache, work, seed, device):
    p.seed_all(seed)
    folder = work / f'seed_{seed}' / 'source'
    folder.mkdir(parents=True, exist_ok=True)
    x_train, y_train, ep_train, x_valid, y_valid, audit = [], [], [], [], [], []
    for subject in base['source_subjects']:
        fit = p.load_recording(subject, 4, base, cache)
        matrix = p.alignment_for(fit, base)
        for run in (4, 8, 12):
            rec = fit if run == 4 else p.load_recording(subject, run, base, cache)
            audit.append(dict(subject=subject, run=run, edf_sha256=rec['source_edf_sha256']))
            if run == 12:
                x_valid.append(apply_alignment(rec['X'], matrix)); y_valid.append(rec['y'])
            else:
                X, y, episodes, _ = dense_examples(rec)
                x_train.append(apply_alignment(X, matrix)); y_train.append(y); ep_train.append(episodes)
        print(json.dumps(dict(prepared_subject=subject)), flush=True)
    X, y, episodes = np.concatenate(x_train), np.concatenate(y_train), np.concatenate(ep_train)
    V, vy = np.concatenate(x_valid), np.concatenate(y_valid)
    identity = p.config_digest(dict(preprocessing=base['preprocessing'], seed=seed, source_audit=audit,
                                    dense_window_s=2., stride_s=.25, label='endpoint', epochs=EPOCHS,
                                    optimizer_updates_per_epoch='cue-window baseline update count'))
    ckpt = folder / 'resume.pt'
    model = model_for(base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    start_epoch, best_loss, bad = 0, float('inf'), 0
    histories = []
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        if saved['identity'] != identity: raise ValueError('resume provenance changed')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        start_epoch, best_loss, bad, histories = saved['epoch'], saved['best_loss'], saved['bad'], saved['histories']
    train_updates = sum(cue_updates(p.load_recording(s, r, base, cache)['X'], SOURCE_BATCH)
                       for s in base['source_subjects'] for r in (4, 8))
    # Match optimizer steps to the baseline source recipe: one epoch-equivalent
    # consumes the same number of cue-window batches from the whole cohort.
    steps_per_epoch = train_updates
    for epoch in range(start_epoch, EPOCHS):
        hist = train_partition(model, opt, X, y, episodes, steps_per_epoch, SOURCE_BATCH,
                               seed + epoch, device, f'source_epoch_{epoch+1}')
        model.eval()
        probs = p.predict(model, V, device)
        val = float(-np.log(probs[np.arange(len(vy)), vy].clip(1e-12)).mean())
        best_loss, bad = (val, 0) if val < best_loss else (best_loss, bad + 1)
        histories.extend(hist)
        state = dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch+1,
                     best_loss=best_loss, bad=bad, validation_loss=val, histories=histories,
                     audit=audit, seed=seed, updates_per_epoch=steps_per_epoch)
        torch.save(state, ckpt)
        if val <= best_loss: torch.save(state, folder / 'best.pt')
        print(json.dumps(dict(seed=seed, source_epoch=epoch+1, validation_loss=val, best_loss=best_loss,
                              bad_epochs=bad, gpu_mem_peak_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
        if bad >= 60: break
    return folder / 'best.pt'


def train_targets(base, cache, work, seed, source_path, device):
    source = torch.load(source_path, map_location=device, weights_only=False)
    source_sha = digest(source_path)
    results = []
    for subject in base['target_subjects']:
        p.seed_all(seed + 100000 * subject)
        folder = work / f'seed_{seed}' / 'targets' / f'S{subject:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        fit = p.load_recording(subject, 4, base, cache)
        dev = p.load_recording(subject, 8, base, cache)
        matrix = p.alignment_for(fit, base)
        X, y, episodes, _ = dense_examples(fit)
        X = apply_alignment(X, matrix)
        # Preserve the saved R04 cue-window optimizer-update budget.
        cue_x = apply_alignment(fit['X'], matrix)
        updates = cue_updates(cue_x, TARGET_BATCH)
        identity = p.config_digest(dict(subject=subject, seed=seed, r04_sha256=fit['source_edf_sha256'],
                                        r08_sha256=dev['source_edf_sha256'], source_sha256=source_sha,
                                        dense_window_s=2., stride_s=.25, label='endpoint', epochs=EPOCHS,
                                        target_lr=.0001, updates_per_epoch=updates))
        ckpt = folder / 'resume.pt'
        model = model_for(base).to(device); model.load_state_dict(source['model'])
        opt = torch.optim.Adam(model.parameters(), lr=.0001, weight_decay=1e-4)
        start_epoch, histories = 0, []
        if ckpt.exists():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            if state['identity'] != identity: raise ValueError('target resume provenance changed')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            start_epoch, histories = state['epoch'], state['histories']
        for epoch in range(start_epoch, EPOCHS):
            hist = train_partition(model, opt, X, y, episodes, updates, TARGET_BATCH,
                                   seed + 100000 * subject + epoch, device, f'S{subject:03d}_target_epoch_{epoch+1}')
            histories.extend(hist)
            state = dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch+1,
                         histories=histories, alignment=torch.as_tensor(matrix), source_sha256=source_sha,
                         r04_sha256=fit['source_edf_sha256'], updates_per_epoch=updates)
            if (epoch+1) % 10 == 0 or epoch+1 == EPOCHS: torch.save(state, ckpt)
        model.eval()
        metric = development(model, dev, matrix, base, device)
        rec = dict(subject=subject, seed=seed, source_sha256=source_sha, r04_sha256=fit['source_edf_sha256'],
                   r08_sha256=dev['source_edf_sha256'], epoch=EPOCHS, updates_per_epoch=updates,
                   model_sha256=digest(ckpt), role='exploratory R08 development; not independent confirmation',
                   metrics=metric)
        (folder / 'result.json').write_text(json.dumps(rec, indent=2) + '\n')
        results.append(rec)
        print(json.dumps(dict(subject=subject, seed=seed, r08_balanced_accuracy=metric['window_metrics']['balanced_accuracy'],
                              command_metrics=metric['command_metrics'])), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = ap.parse_args()
    base = p.read_config(args.config)
    p.validate_protocol(base)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA device not available; refusing CPU training')
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(device=torch.cuda.get_device_name(device), total_vram=torch.cuda.get_device_properties(device).total_memory,
                          config_sha256=digest(args.config), code_sha256=digest(__file__), seeds=args.seeds)), flush=True)
    for seed in args.seeds:
        source_path = fit_source(base, args.cache, args.work, seed, device)
        results = train_targets(base, args.cache, args.work, seed, source_path, device)
        out = args.work / f'seed_{seed}' / 'target_summary.json'
        out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__': main()
