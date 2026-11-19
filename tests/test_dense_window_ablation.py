import numpy as np
import pytest
import torch

from eeg_decoder import dense_window_ablation as dense


def test_dense_source_uses_pooled_cue_count(tmp_path, monkeypatch):
    # Four recordings with three cues each fill a five-example batch twice,
    # followed by a two-example batch. Per-record rounding would take four steps.
    base = dense.p.read_config('configs/gpu_research.json')
    base['source_subjects'] = [16, 17]
    base['alignment'] = 'none'
    seen_batches = []

    def record(subject, run, *_):
        return dict(subject=subject, run=run, source_edf_sha256=f'{subject}-{run}',
                    signal=np.ones((9, 800), dtype=np.float32),
                    X=np.ones((3, 9, 320), dtype=np.float32), y=np.array([0, 1, 0]),
                    intervals=[(0., 2.5, 0), (2.5, 5., 1)])

    def capture_batch(model, x, y, optimizer, indices, device):
        seen_batches.append(len(indices))
        return 0.

    monkeypatch.setattr(dense, 'EPOCHS', 1)
    monkeypatch.setattr(dense, 'SOURCE_BATCH', 5)
    monkeypatch.setattr(dense.p, 'load_recording', record)
    monkeypatch.setattr(dense, 'update', capture_batch)
    monkeypatch.setattr(dense.p, 'predict', lambda model, x, dev: np.full((len(x), 2), .5))
    best = dense.fit_source(base, tmp_path, tmp_path, 42, torch.device('cpu'))
    assert seen_batches == [5, 5, 2]
    state = torch.load(best, weights_only=True)
    assert state['samples_per_epoch'] == 12
    assert state['updates_per_epoch'] == 3
    # The original checkpoint lacks the new budget identity.
    state['identity'] = 'historical'
    torch.save(state, best.parent / 'resume.pt')
    with pytest.raises(ValueError, match='provenance'):
        dense.fit_source(base, tmp_path, tmp_path, 42, torch.device('cpu'))


def test_odd_target_batch_spends_full_budget_and_balances_classes():
    labels = np.array([0, 0, 1, 1])
    batches = list(dense.episode_sampler(labels, np.arange(4), 30, 15, 42))
    assert [len(batch) for batch in batches] == [15, 15]
    assert sorted(labels[batch].sum() for batch in batches) == [7, 8]
