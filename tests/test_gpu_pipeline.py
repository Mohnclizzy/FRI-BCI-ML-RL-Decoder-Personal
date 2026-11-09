"""Unit tests for Euclidean alignment, EEGNet, HysteresisGate, and GPU pipeline split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='Install requirements-gpu.txt for GPU pipeline tests.')

import eeg_decoder.gpu_pipeline as pipeline

from eeg_decoder.alignment import apply_alignment, fit_alignment
from eeg_decoder.gpu_pipeline import (
    adapt,
    calibrate,
    config_digest,
    digest,
    evaluate,
    pretrain,
    read_config,
    validate_protocol,
)
from eeg_decoder.neural import EEGNet
from eeg_decoder.risk_gate import HysteresisGate, calibrate_threshold, rest_episode_maxima


def test_euclidean_alignment_whitens_mean_covariance() -> None:
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((16, 9, 320))
    mixing = rng.standard_normal((9, 9)) + 2.0 * np.eye(9)
    windows = np.einsum("cd,ndt->nct", mixing, raw)
    matrix = fit_alignment(windows)
    aligned = apply_alignment(windows, matrix)
    centered = aligned - aligned.mean(axis=-1, keepdims=True)
    cov = np.einsum("nct,ndt->cd", centered, centered) / (len(aligned) * (aligned.shape[-1] - 1))
    assert aligned.shape == windows.shape
    assert np.allclose(cov, np.eye(9), atol=1e-4)


def test_eegnet_forward_and_weight_constraints() -> None:
    model = EEGNet(channels=9, samples=320)
    with torch.no_grad():
        model.features[2].weight.mul_(10.0)
        model.classifier.weight.mul_(10.0)
    model.constrain_weights()
    spatial_norms = model.features[2].weight.flatten(1).norm(dim=1)
    cls_norms = model.classifier.weight.flatten(1).norm(dim=1)
    assert torch.all(spatial_norms <= 1.0 + 1e-5)
    assert torch.all(cls_norms <= 0.25 + 1e-5)
    x = torch.randn(4, 9, 320)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_hysteresis_gate_and_crc_calibration() -> None:
    scores = np.array([0.52, 0.58, 0.61, 0.64, 0.68, 0.71, 0.74, 0.79, 0.81, 0.84, 0.90])
    cal = calibrate_threshold(scores, alpha=0.2)
    assert cal["status"] == "empirical_crc_candidate"
    assert 0.80 <= cal["threshold"] <= 0.90
    assert cal["guarantee_claimed"] is False

    gate = HysteresisGate(threshold=0.70, margin=0.05, persistence_s=0.5, max_age_s=1.0)
    # Below enable threshold (0.75) stays in hold
    d0 = gate.update(0.72, 2.0, 2.0, 2.0, "S001R12")
    assert d0["skill"] == "hold" and d0["reason"] == "below_enable_threshold"
    # Meets enable threshold (0.75) starts collecting evidence
    d1 = gate.update(0.78, 2.25, 2.25, 2.25, "S001R12")
    assert d1["skill"] == "hold" and d1["reason"] == "collecting_evidence"
    # After 0.5s persistence, enables grasp_lift
    d2 = gate.update(0.76, 2.75, 2.75, 2.75, "S001R12")
    assert d2["skill"] == "grasp_lift" and d2["state"] == "enable"
    # Hysteresis band: 0.72 >= threshold (0.70) keeps enable active!
    d3 = gate.update(0.72, 3.00, 3.00, 3.00, "S001R12")
    assert d3["skill"] == "grasp_lift" and d3["state"] == "enable"
    # Dropping below hold threshold (0.70) returns to hold
    d4 = gate.update(0.68, 3.25, 3.25, 3.25, "S001R12")
    assert d4["skill"] == "hold" and d4["reason"] == "below_hold_threshold"


def test_rest_episode_maxima_and_protocol_validation() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0])
    probs = np.array([0.8, 0.4, 0.9, 0.3])
    intervals = [[0.0, 1.5, 0], [1.5, 3.2, 1], [3.2, 5.0, 0]]
    # Episode 1 [0, 1.5] overlaps t=1.0 (0.8); Episode 2 [3.2, 5.0] overlaps carried t=3.0 (0.9) and t=4.0 (0.3)
    maxima = rest_episode_maxima(probs, times, intervals, max_age_s=1.0)
    assert np.allclose(maxima, [0.8, 0.9])

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "gpu_research.json"
    cfg = read_config(cfg_path)
    assert len(config_digest(cfg)) == 64
    bad_cfg = dict(cfg, source_subjects=[1, 16])
    with pytest.raises(ValueError, match="disjoint"):
        validate_protocol(bad_cfg)


def test_end_to_end_stages_strict_partition(tmp_path: Path, monkeypatch) -> None:
    cfg = read_config(Path(__file__).resolve().parents[1] / "configs" / "gpu_research.json")
    cfg = {
        **cfg,
        "source_subjects": [16],
        "target_subjects": [1],
        "pretrain_epochs": 2,
        "adapt_epochs": 2,
        "batch_size": 4,
        "patience": 2,
    }
    validate_protocol(cfg)
    p = cfg["preprocessing"]
    key = config_digest(p)[:16]
    cache = tmp_path / "cache" / key
    cache.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    for subject in (16, 1):
        for run in (4, 8, 12):
            stem = f"S{subject:03d}R{run:02d}"
            X = rng.standard_normal((6, 9, 320)).astype(np.float64)
            y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
            signal = rng.standard_normal((9, 160 * 12)).astype(np.float64)
            npz_path = cache / f"{stem}.npz"
            np.savez_compressed(npz_path, X=X, y=y, signal=signal)
            meta = {
                "subject": subject,
                "run": run,
                "preprocessing": p,
                "records": [{"trial_id": i, "label_id": int(y[i])} for i in range(len(y))],
                "skipped": [],
                "source_edf_sha256": "0" * 64,
                "npz_sha256": digest(npz_path),
                "class_counts": [3, 3],
                "intervals": [[0.0, 4.0, 0], [4.0, 8.0, 1], [8.0, 12.0, 0]],
            }
            (cache / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")

    # A forbidden run access must fail even if all six cached files exist.
    original_loader = pipeline.load_recording
    allowed = {(16, 4), (16, 8), (16, 12)}
    calls = []
    def guarded_loader(subject, run, *args, **kwargs):
        assert (subject, run) in allowed, 'Premature evaluation/calibration data access'
        calls.append((subject, run))
        return original_loader(subject, run, *args, **kwargs)
    monkeypatch.setattr(pipeline, 'load_recording', guarded_loader)
    device = torch.device("cpu")
    best_ckpt = pretrain(cfg, tmp_path, device)
    assert best_ckpt.is_file()
    assert set(calls) == allowed
    before_resume = digest(best_ckpt)
    pretrain(cfg, tmp_path, device, resume=True)
    assert digest(best_ckpt) == before_resume

    allowed = {(1, 4)}; calls.clear()
    model_path = adapt(cfg, tmp_path, device, subject=1)
    assert model_path.is_file()
    assert set(calls) == allowed
    adapt(cfg, tmp_path, device, subject=1, resume=True)

    allowed = {(1, 8)}; calls.clear()
    cal_path = calibrate(cfg, tmp_path, device, subject=1)
    cal_data = json.loads(cal_path.read_text(encoding="utf-8"))
    assert cal_data["calibration_run"] == 8
    assert cal_data["guarantee_claimed"] is False
    assert set(calls) == allowed
    with pytest.raises(FileExistsError):
        calibrate(cfg, tmp_path, device, subject=1)

    allowed = {(1, 12)}; calls.clear()
    eval_dir = evaluate(cfg, tmp_path, device, subject=1)
    decisions_file = eval_dir / "S001R12-decisions.jsonl"
    metrics_file = eval_dir / "metrics.json"
    assert decisions_file.is_file() and metrics_file.is_file()
    first_row = json.loads(decisions_file.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_row["decisions"].keys()) == {"raw", "confidence", "temporal", "crc_hysteresis"}
    assert "label" not in first_row["estimate"]
    assert set(calls) == allowed
    with pytest.raises(FileExistsError):
        evaluate(cfg, tmp_path, device, subject=1)

    # Fine-tuning and scratch controls must share exactly the same fit IDs.
    allowed = {(1, 4)}
    scratch = adapt(cfg, tmp_path, device, subject=1, scratch=True)
    a = json.loads((scratch.parent / 'adaptation.json').read_text())
    b = json.loads((model_path.parent / 'adaptation.json').read_text())
    assert a['fit_trial_ids'] == b['fit_trial_ids']
    assert a['pretrain_sha256'] is None

    # A changed model cannot reuse an old calibrated threshold.
    saved = torch.load(model_path, weights_only=True)
    saved['model']['classifier.bias'] += 0.1
    torch.save(saved, model_path)
    with pytest.raises(ValueError, match='Calibration does not match'):
        evaluate(cfg, tmp_path, device, subject=1)

    # A valid cache with a changed source EDF identity still invalidates resume.
    meta_path = cache / 'S016R04.json'
    meta = json.loads(meta_path.read_text()); meta['source_edf_sha256'] = '1' * 64
    meta_path.write_text(json.dumps(meta))
    allowed = {(16, 4), (16, 8), (16, 12)}
    with pytest.raises(ValueError, match='Source data changed'):
        pretrain(cfg, tmp_path, device, resume=True)


def test_crc_small_sample_fallback_and_threshold_ties():
    result = calibrate_threshold([0.1, 0.2], alpha=0.1)
    assert result['threshold'] > 1 and result['status'] == 'always_hold_fallback'
    gate = HysteresisGate(result['threshold'], persistence_s=0)
    assert gate.update(1.0, 1, 1, 1, 'test')['skill'] == 'hold'
    assert calibrate_threshold(np.ones(20), alpha=0.1)['status'] == 'always_hold_fallback'
    result = calibrate_threshold(np.repeat(0.7, 20), alpha=0.1)
    assert result['threshold'] > 0.7
    assert result['selected_crc_adjusted_risk'] <= 0.1


def test_gate_future_stale_duplicate_and_episode_reset():
    gate = HysteresisGate(0.7, persistence_s=0.5)
    assert gate.update(1, 1, 2, 1, 'a')['reason'] == 'not_yet_available'
    assert gate.update(1, 1, 1, 3, 'a')['reason'] == 'stale'
    gate.update(1, 3, 3, 3, 'a')
    assert gate.update(1, 3.5, 3.5, 3.5, 'a')['skill'] == 'grasp_lift'
    assert gate.tick(4.6) == 'hold'
    assert gate.update(1, 3.5, 3.5, 4, 'a')['reason'] == 'duplicate_or_out_of_order'
    gate.update(1, 5, 5, 5, 'a')
    assert gate.update(1, 5.5, 5.5, 5.5, 'b')['skill'] == 'hold'
    with pytest.raises(ValueError):
        gate.update(float('nan'), 6, 6, 6, 'b')
    with pytest.raises(ValueError):
        gate.update(1, 6, 5, 6, 'b')


def test_rest_envelope_bounds_hysteresis_activation():
    rng = np.random.default_rng(93)
    times = np.arange(0.25, 12, 0.25)
    probs = rng.uniform(size=len(times))
    intervals = [[0.8, 2.7, 0], [4.1, 8.3, 0], [9.2, 11.9, 0]]
    maxima = rest_episode_maxima(probs, times, intervals, 1.0)
    for threshold in (0.5, 0.7, 0.9):
        gate = HysteresisGate(threshold, persistence_s=0.25)
        skills = [gate.update(float(p), float(t), float(t), float(t), 'x')['skill'] for p, t in zip(probs, times)]
        measured = pipeline.command_metrics(times, skills, intervals, 1.0)
        assert measured['rest_episodes_with_any_command'] <= int((maxima >= threshold).sum())
    # An expired score cannot count as carried activation in a later episode.
    assert rest_episode_maxima([0.99], [1.0], [[3, 4, 0]], 1.0).tolist() == [0.0]


def test_rank_deficient_alignment_and_invalid_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 9, 320)); X -= X.mean(axis=1, keepdims=True)
    assert np.isfinite(apply_alignment(X, fit_alignment(X))).all()
    with pytest.raises(ValueError, match='Zero-variance'):
        fit_alignment(np.zeros_like(X))
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_alignment(X)


def test_annotation_mapping_and_control_metrics():
    annotations = [{'onset': i * 4, 'duration': 4, 'description': code} for i, code in enumerate(['T0', 'T1', 'T2'])]
    assert pipeline.annotations_to_intervals(annotations, 12) == [[0, 4, 0], [4, 8, 1], [8, 12, 1]]
    metrics = pipeline.command_metrics([1, 3], ['grasp_lift', 'hold'], [[0, 4, 0]], max_age_s=1)
    assert metrics['rest_command_seconds'] == 1
    assert metrics['rest_command_fraction'] == 0.25


def test_notebook_python_cells_compile():
    path = Path(__file__).resolve().parents[1] / 'notebooks/FRI_BCI_Colab_Master.ipynb'
    notebook = json.loads(path.read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{i}', 'exec')

