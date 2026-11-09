"""EA-EEGNet research stages. Run `python -m eeg_decoder.gpu_pipeline --help`.

Target R12 is opened only by the explicit evaluate stage. Model fitting and EA
use target R04; gate calibration uses R08. No downloads occur on import.
"""
import argparse
import hashlib
import json
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import mne
import numpy as np
import torch
from mne.datasets import eegbci
from torch.utils.data import DataLoader, TensorDataset

from .alignment import apply_alignment, fit_alignment
from .baseline import extract_windows, preprocess_run, score, validate_config
from .neural import EEGNet
from .risk_gate import HysteresisGate, calibrate_threshold, rest_episode_maxima


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_protocol(config)
    return config


def validate_protocol(c):
    for key in ('source_subjects', 'target_subjects'):
        subjects = c[key]
        if not subjects or len(set(subjects)) != len(subjects) or any(type(s) is not int or not 1 <= s <= 109 for s in subjects):
            raise ValueError('Invalid or duplicated subject IDs: ' + key)
    if set(c['source_subjects']) & set(c['target_subjects']):
        raise ValueError('Source and target subjects must be disjoint.')
    excluded = {int(s) for s in c.get('excluded_source_subjects_note', {})}
    if excluded & set(c['source_subjects']):
        raise ValueError('A source subject is both included and excluded.')
    for key in ('pretrain_lr', 'adapt_lr'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Learning rates must be finite and positive.')
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**31:
        raise ValueError('seed must be a nonnegative 31-bit integer.')
    if c['source_train_runs'] != [4, 8] or c['source_validation_run'] != 12:
        raise ValueError('Source protocol is R04/R08 fit and R12 validation.')
    if (c['target_fit_run'], c['target_calibration_run'], c['target_test_run']) != (4, 8, 12):
        raise ValueError('Target fit/calibration/evaluation must be separate runs 4/8/12.')
    if c['alignment'] not in ('ea', 'none'):
        raise ValueError('alignment must be ea or none.')
    p = c['preprocessing']
    validate_config({**p, 'subjects': c['target_subjects'], 'runs': [4, 8, 12], 'test_subjects': [], 'test_run': 12})
    if p['sample_rate_hz'] != 160 or p['window_seconds'] * 160 < 32:
        raise ValueError('This protocol requires 160 Hz and at least 32 samples/window.')
    for key in ('pretrain_epochs', 'adapt_epochs', 'batch_size', 'patience'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Expected positive integer: ' + key)
    if c['few_shot_per_class'] is not None and (type(c['few_shot_per_class']) is not int or c['few_shot_per_class'] < 1):
        raise ValueError('few_shot_per_class must be positive or null.')
    if not 0 < c['alpha'] < 1 or not 0 < c['stride_s'] <= c['max_age_s']:
        raise ValueError('Invalid risk level or replay stride.')
    if not np.isclose(c['stride_s'] * p['sample_rate_hz'], round(c['stride_s'] * p['sample_rate_hz'])):
        raise ValueError('Replay stride must be an integer number of samples.')
    HysteresisGate(0.5, c['hysteresis_margin'], c['persistence_s'], c['max_age_s'])


def provenance(config):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'config': config, 'config_sha256': config_digest(config), 'git_commit': commit,
            'git_dirty': dirty, 'python': platform.python_version(),
            'versions': {n: version(n) for n in ('torch', 'numpy', 'scipy', 'mne', 'scikit-learn')},
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'claim': 'Implementation run; no guaranteed accuracy, latency, prospective validity, or physical safety.'}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Select GPU runtime or explicitly use --device cpu.')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else 'cpu' if name == 'auto' else name)


def annotations_to_intervals(annotations, duration):
    result = []
    mapping = {'T0': 0, 'T1': 1, 'T2': 1}
    for i, a in enumerate(annotations):
        if a['description'] not in mapping:
            continue
        start = max(0.0, float(a['onset']))
        next_start = annotations[i + 1]['onset'] if i + 1 < len(annotations) else duration
        end = min(duration, next_start, start + a['duration'] if a['duration'] > 0 else duration)
        if end > start:
            result.append([start, float(end), mapping[a['description']]])
    return result


def validate_recording(record, subject, run, preprocessing):
    if record['subject'] != subject or record['run'] != run:
        raise ValueError('Cache subject/run identity mismatch.')
    X, y, signal = record['X'], record['y'], record['signal']
    channels = len(preprocessing['channels'])
    width = round(preprocessing['window_seconds'] * preprocessing['sample_rate_hz'])
    if X.ndim != 3 or X.shape[1:] != (channels, width) or y.shape != (len(X),):
        raise ValueError('Cache trial shape mismatch.')
    if signal.ndim != 2 or signal.shape[0] != channels or signal.shape[1] < width:
        raise ValueError('Cache signal shape mismatch.')
    if not np.isfinite(X).all() or not np.isfinite(signal).all() or set(y.tolist()) != {0, 1}:
        raise ValueError('Invalid cached samples or labels.')
    if len(record['records']) != len(y) or any(r['label_id'] != int(label) for r, label in zip(record['records'], y)):
        raise ValueError('Cache metadata/label mismatch.')
    return record


def load_recording(subject, run, config, work):
    """Cache complete causal signals plus independent annotation windows.

    Known quality problems fail explicitly; no silent participant exclusions or
    resampling. A changed preprocessing config gets a different cache key.
    """
    p = config['preprocessing']
    key = config_digest(p)[:16]
    cache = Path(work) / 'cache' / key
    cache.mkdir(parents=True, exist_ok=True)
    stem = f'S{subject:03d}R{run:02d}'
    arrays, metadata = cache / (stem + '.npz'), cache / (stem + '.json')
    if arrays.exists() and metadata.exists():
        meta = json.loads(metadata.read_text())
        if meta['preprocessing'] != p or meta['npz_sha256'] != digest(arrays):
            raise ValueError('Cache integrity failure: ' + stem)
        with np.load(arrays, allow_pickle=False) as saved:
            return validate_recording({**meta, **{k: saved[k] for k in ('X', 'y', 'signal')}}, subject, run, p)
    data_path = Path(work) / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    filenames = eegbci.load_data(subject, [run], path=str(data_path.resolve()), update_path=False, verbose='ERROR')
    if len(filenames) != 1:
        raise ValueError('Expected exactly one EDF: ' + stem)
    with mne.io.read_raw_edf(filenames[0], preload=True, verbose='ERROR') as raw:
        eegbci.standardize(raw)
        if not np.isclose(raw.info['sfreq'], p['sample_rate_hz']):
            raise ValueError(stem + ': sampling-rate mismatch; explicitly review eligibility before changing config.')
        raw.pick(p['channels']); raw.reorder_channels(p['channels'])
        signal = preprocess_run(raw.get_data(), p)
        annotations = [{'onset': float(onset), 'duration': float(duration), 'description': str(description)}
                       for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)]
    X, records, skipped = extract_windows(signal, annotations, subject, run, p)
    y = np.asarray([r['label_id'] for r in records], dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError(stem + ': both REST and imagery are required.')
    np.savez_compressed(arrays, X=X, y=y, signal=signal)
    meta = {'subject': subject, 'run': run, 'preprocessing': p, 'records': records, 'skipped': skipped,
            'source_edf_sha256': digest(filenames[0]), 'npz_sha256': digest(arrays),
            'class_counts': np.bincount(y, minlength=2).tolist(),
            'intervals': annotations_to_intervals(annotations, signal.shape[1] / p['sample_rate_hz'])}
    write_json(metadata, meta)
    return validate_recording({**meta, 'X': X, 'y': y, 'signal': signal}, subject, run, p)


def alignment_for(recording, config):
    if config['alignment'] == 'ea':
        return fit_alignment(recording['X'])
    # The no-EA ablation uses a fixed volts-to-microvolts conversion, never test statistics.
    return np.eye(recording['X'].shape[1]) * 1e6


def make_model(c):
    return EEGNet(channels=len(c['preprocessing']['channels']),
                  samples=round(c['preprocessing']['window_seconds'] * c['preprocessing']['sample_rate_hz']))


def predict(model, X, device, batch_size=256):
    model.eval(); chunks = []
    with torch.no_grad():
        for begin in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(np.asarray(X[begin:begin+batch_size], dtype=np.float32)).to(device))
            chunks.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(chunks)


def train_epoch(model, X, y, device, optimizer, c, seed):
    seed_all(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=c['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    model.train(); total = 0.0
    for data, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(data.to(device)), labels.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); model.constrain_weights()
        total += loss.item() * len(labels)
    return total / len(y)


def save_checkpoint(path, value):
    temp = Path(str(path) + '.partial')
    torch.save(value, temp); temp.replace(path)


def assert_checkpoint(checkpoint, config):
    if checkpoint['config_sha256'] != config_digest(config):
        raise ValueError('Checkpoint config mismatch; use a new work directory for a new protocol.')


def pretrain(c, work, device, resume=False):
    output = Path(work) / 'pretrain'
    output.mkdir(parents=True, exist_ok=True)
    last = output / 'last.pt'
    if (output / 'manifest.json').exists() and not resume:
        raise FileExistsError('Pretraining already started; use --resume or a new work directory.')
    if (output / 'manifest.json').exists():
        assert_checkpoint(json.loads((output / 'manifest.json').read_text()), c)
    write_json(output / 'manifest.json', provenance(c))
    train_x, train_y, valid_x, valid_y, audit = [], [], [], [], []
    for subject in c['source_subjects']:
        fitting = load_recording(subject, 4, c, work)
        matrix = alignment_for(fitting, c)
        np.save(output / f'S{subject:03d}-alignment.npy', matrix)
        for run in (4, 8, 12):
            record = fitting if run == 4 else load_recording(subject, run, c, work)
            X = apply_alignment(record['X'], matrix)
            xs, ys = (valid_x, valid_y) if run == 12 else (train_x, train_y)
            xs.append(X); ys.append(record['y'])
            audit.append({'subject': subject, 'run': run, 'role': 'source_validation' if run == 12 else 'source_train',
                          'class_counts': record['class_counts'], 'source_edf_sha256': record['source_edf_sha256'],
                          'trial_ids': [r['trial_id'] for r in record['records']]})
        print(f'Prepared source S{subject:03d}', flush=True)
    X, y, V, v = map(np.concatenate, (train_x, train_y, valid_x, valid_y))
    data_manifest = {'recordings': audit, 'training_windows': len(y), 'validation_windows': len(v)}
    data_sha = config_digest(data_manifest)
    seed_all(c['seed']); model = make_model(c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['pretrain_lr'], weight_decay=1e-4)
    start, best, bad, history = 0, float('inf'), 0, []
    if resume and last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=True); assert_checkpoint(checkpoint, c)
        if checkpoint['data_manifest_sha256'] != data_sha:
            raise ValueError('Source data changed since the checkpoint; refusing resume.')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start, best, bad, history = checkpoint['epoch'] + 1, checkpoint['best'], checkpoint['bad'], checkpoint['history']
    write_json(output / 'data_manifest.json', data_manifest)
    for epoch in range(start, c['pretrain_epochs']):
        if bad >= c['patience']:
            break
        loss = train_epoch(model, X, y, device, optimizer, c, c['seed'] + epoch)
        probs = predict(model, V, device)
        val_loss = float(-np.log(np.maximum(probs[np.arange(len(v)), v], 1e-12)).mean())
        improved = val_loss < best
        best, bad = (val_loss, 0) if improved else (best, bad + 1)
        history.append({'epoch': epoch + 1, 'train_loss': loss, 'source_validation_loss': val_loss})
        saved = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'best': best, 'bad': bad, 'history': history, 'config_sha256': config_digest(c),
                 'data_manifest_sha256': data_sha}
        if improved:
            save_checkpoint(output / 'best.pt', saved)
        save_checkpoint(last, saved); write_json(output / 'history.json', history)
        print(history[-1], flush=True)
    return output / 'best.pt'


def adapt(c, work, device, subject, resume=False, scratch=False):
    if subject not in c['target_subjects']:
        raise ValueError('Not a declared target subject.')
    kind = 'scratch' if scratch else 'transfer'
    output = Path(work) / kind / f'S{subject:03d}'
    model_path = output / 'model.pt'
    if model_path.exists():
        if not resume:
            raise FileExistsError('Target model exists; use --resume or a new work directory.')
        saved = torch.load(model_path, map_location='cpu', weights_only=True); assert_checkpoint(saved, c)
        if not scratch and saved['pretrain_sha256'] != digest(Path(work) / 'pretrain/best.pt'):
            raise ValueError('Pretrained checkpoint changed after adaptation.')
        return model_path
    output.mkdir(parents=True, exist_ok=True)
    record = load_recording(subject, 4, c, work)
    matrix = alignment_for(record, c)
    indices = np.arange(len(record['y']))
    if c['few_shot_per_class'] is not None:
        rng = np.random.default_rng(c['seed'] + subject)
        chosen = []
        for label in (0, 1):
            candidates = indices[record['y'] == label]
            if len(candidates) < c['few_shot_per_class']:
                raise ValueError('Insufficient R04 trials for the requested few-shot count.')
            chosen.extend(rng.choice(candidates, c['few_shot_per_class'], replace=False).tolist())
        indices = np.sort(chosen)
    X, y = apply_alignment(record['X'][indices], matrix), record['y'][indices]
    seed_all(c['seed'] + subject); model = make_model(c).to(device)
    pretrain_sha = None
    if not scratch:
        path = Path(work) / 'pretrain/best.pt'
        saved = torch.load(path, map_location=device, weights_only=True); assert_checkpoint(saved, c)
        model.load_state_dict(saved['model']); pretrain_sha = digest(path)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['adapt_lr'], weight_decay=1e-4)
    history = []
    for epoch in range(c['adapt_epochs']):
        loss = train_epoch(model, X, y, device, optimizer, c, c['seed'] + 1000 * subject + epoch)
        history.append({'epoch': epoch + 1, 'fit_loss': loss})
    save_checkpoint(model_path, {'model': model.state_dict(), 'alignment': torch.from_numpy(matrix),
                                'config_sha256': config_digest(c), 'pretrain_sha256': pretrain_sha,
                                'subject': subject, 'fit_run': 4, 'source_edf_sha256': record['source_edf_sha256']})
    write_json(output / 'adaptation.json', {**provenance(c), 'subject': subject, 'kind': kind,
               'model_sha256': digest(model_path), 'pretrain_sha256': pretrain_sha,
               'fit_trial_ids': [record['records'][i]['trial_id'] for i in indices],
               'alignment_trial_ids': [r['trial_id'] for r in record['records']],
               'source_edf_sha256': record['source_edf_sha256'], 'history': history,
               'note': 'Few-shot count limits labels for gradients; alignment uses all unlabeled R04 windows. No R08/R12 fitting.'})
    return model_path


def load_target(c, work, device, subject, scratch=False):
    kind = 'scratch' if scratch else 'transfer'
    output = Path(work) / kind / f'S{subject:03d}'
    path = output / 'model.pt'
    saved = torch.load(path, map_location=device, weights_only=True); assert_checkpoint(saved, c)
    if saved['subject'] != subject or saved['fit_run'] != 4:
        raise ValueError('Wrong subject or fit partition in target checkpoint.')
    model = make_model(c).to(device); model.load_state_dict(saved['model'])
    return model, saved['alignment'].cpu().numpy(), path


def replay_probabilities(model, record, matrix, c, device):
    """Sliding causal windows use signal samples only, never annotation timing."""
    sfreq = c['preprocessing']['sample_rate_hz']
    width = round(c['preprocessing']['window_seconds'] * sfreq)
    step = round(c['stride_s'] * sfreq)
    signal = apply_alignment(record['signal'], matrix)
    ends = np.arange(width, signal.shape[1] + 1, step)
    if not len(ends):
        raise ValueError('Recording too short for replay.')
    windows = np.stack([signal[:, end-width:end] for end in ends])
    return predict(model, windows, device)[:, 1], ends / sfreq


def calibrate(c, work, device, subject, scratch=False):
    model, matrix, path = load_target(c, work, device, subject, scratch)
    destination = path.parent / 'calibration.json'
    if destination.exists():
        raise FileExistsError('Calibration is frozen; use a new work directory to change it.')
    record = load_recording(subject, 8, c, work)
    probabilities, times = replay_probabilities(model, record, matrix, c, device)
    maxima = rest_episode_maxima(probabilities, times, record['intervals'], c['max_age_s'])
    result = calibrate_threshold(maxima, c['alpha'])
    result.update({'config_sha256': config_digest(c), 'model_sha256': digest(path), 'subject': subject,
                   'calibration_run': 8, 'source_edf_sha256': record['source_edf_sha256'],
                   'episode_maxima': maxima.tolist(), 'hysteresis_margin': c['hysteresis_margin'],
                   'persistence_s': c['persistence_s'], 'max_age_s': c['max_age_s']})
    write_json(destination, result)
    return destination


def command_metrics(times, skills, intervals, max_age_s):
    times = np.asarray(times)
    ends = np.minimum(np.r_[times[1:], np.inf], times + max_age_s)
    enabled = np.asarray(skills) == 'grasp_lift'
    totals = {0: 0.0, 1: 0.0}; active = {0: 0.0, 1: 0.0}; rest_events = 0; rest_count = 0
    for start, end, label in intervals:
        overlap = np.maximum(0.0, np.minimum(ends, end) - np.maximum(times, start))
        seconds = float(overlap[enabled].sum())
        totals[label] += end - start; active[label] += seconds
        if label == 0:
            rest_count += 1; rest_events += int(seconds > 0)
    return {'rest_command_seconds': active[0], 'rest_seconds': totals[0],
            'rest_command_fraction': active[0] / totals[0] if totals[0] else None,
            'imagery_command_seconds': active[1], 'imagery_seconds': totals[1],
            'missed_imagery_fraction': 1 - active[1] / totals[1] if totals[1] else None,
            'rest_episodes_with_any_command': rest_events, 'rest_episode_count': rest_count,
            'rest_episode_activation_fraction': rest_events / rest_count if rest_count else None,
            'physical_robot_metrics': 'not measured; these are command-stream metrics only'}


def evaluate(c, work, device, subject, scratch=False):
    model, matrix, path = load_target(c, work, device, subject, scratch)
    calibration_path = path.parent / 'calibration.json'
    calibration = json.loads(calibration_path.read_text()); assert_checkpoint(calibration, c)
    if calibration['model_sha256'] != digest(path) or calibration['subject'] != subject or calibration['calibration_run'] != 8:
        raise ValueError('Calibration does not match the frozen target model.')
    destination = path.parent / 'evaluation'
    if destination.exists():
        raise FileExistsError('Evaluation output already exists; use a new work directory.')
    record = load_recording(subject, 12, c, work)
    probabilities, times = replay_probabilities(model, record, matrix, c, device)
    gates = {
        'raw': HysteresisGate(float(np.nextafter(0.5, 1)), 0, 0, c['max_age_s']),
        'confidence': HysteresisGate(0.7, 0, 0, c['max_age_s']),
        'temporal': HysteresisGate(0.7, 0, 0.5, c['max_age_s']),
        'crc_hysteresis': HysteresisGate(calibration['threshold'], c['hysteresis_margin'], c['persistence_s'], c['max_age_s']),
    }
    commands = {name: [] for name in gates}
    source = f'S{subject:03d}R12'; rows = []
    for probability, time in zip(probabilities, times):
        decisions = {name: gate.update(float(probability), float(time), float(time), float(time), source) for name, gate in gates.items()}
        for name in gates:
            commands[name].append(decisions[name]['skill'])
        rows.append({'estimate': {'source': source, 'episode': source, 'window_end_s': float(time),
                                 'available_at_s': float(time), 'p_rest': float(1-probability), 'p_hand': float(probability)},
                     'decisions': decisions})
    # Annotation labels intentionally excluded from the robot-facing JSONL.
    destination.mkdir(exist_ok=False)
    (destination / (source + '-decisions.jsonl')).write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    window_probs = predict(model, apply_alignment(record['X'], matrix), device)
    metrics = {**provenance(c), 'subject': subject, 'model_sha256': digest(path),
               'calibration_sha256': digest(calibration_path), 'source_edf_sha256': record['source_edf_sha256'],
               'window_metrics': score(record['y'], window_probs.argmax(axis=1)),
               'command_metrics': {name: command_metrics(times, skills, record['intervals'], c['max_age_s']) for name, skills in commands.items()},
               'evaluation_trial_ids': [r['trial_id'] for r in record['records']],
               'probabilities_are_calibrated': False, 'inference_latency_simulated_seconds': 0,
               'claim': 'Personalized held-out recording replay. Previously inspected targets are exploratory; no unopened-cohort claim.'}
    write_json(destination / 'metrics.json', metrics)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['validate', 'pretrain', 'adapt', 'calibrate', 'evaluate'])
    parser.add_argument('--config', default='configs/gpu_research.json')
    parser.add_argument('--work-dir', default='outputs/gpu_research')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--subject', type=int)
    parser.add_argument('--resume', action='store_true', help='Resume pretraining or reuse a verified adapted checkpoint.')
    parser.add_argument('--scratch', action='store_true', help='Train the target EEGNet without source pretraining.')
    args = parser.parse_args(); c = read_config(args.config)
    if args.stage == 'validate':
        print(json.dumps({'source_subjects': len(c['source_subjects']), 'target_subjects': len(c['target_subjects']),
                          'config_sha256': config_digest(c), 'protocol': 'source 4/8 fit, 12 validation; target 4 fit, 8 calibrate, 12 evaluate'}, indent=2)); return
    device = device_for(args.device)
    if args.stage == 'pretrain':
        if args.scratch or args.subject is not None:
            parser.error('--scratch and --subject apply only to target stages.')
        print(pretrain(c, args.work_dir, device, args.resume)); return
    if args.subject not in c['target_subjects']:
        parser.error('--subject must name one configured target participant.')
    if args.resume and args.stage not in ('adapt',):
        parser.error('Calibration/evaluation are immutable; --resume is supported for pretrain/adapt only.')
    if args.stage == 'adapt':
        print(adapt(c, args.work_dir, device, args.subject, args.resume, args.scratch))
    elif args.stage == 'calibrate':
        print(calibrate(c, args.work_dir, device, args.subject, args.scratch))
    else:
        print(evaluate(c, args.work_dir, device, args.subject, args.scratch))


if __name__ == '__main__':
    main()
