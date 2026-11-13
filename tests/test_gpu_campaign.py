import copy
import json

import numpy as np
import pytest
import torch

from eeg_decoder import gpu_campaign as c


def test_campaign_split_resume_and_integrity(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    plan, base = c.read_plan('configs/gpu_campaign.json')
    plan.update(seeds=[42], models={'tiny': dict(f1=2, depth=1, f2=2)}, source_epochs=2,
                source_patience=2, target_epochs=[1, 2], target_lrs=[.001], checkpoint_every=1)
    base.update(source_subjects=[16], target_subjects=[1])
    accesses = []
    def record(subject, run, config, work):
        accesses.append((subject, run))
        assert subject != 1 or run in (4, 8), 'Target R12 was accessed'
        rng = np.random.default_rng(subject * 100 + run)
        return {'X': rng.normal(size=(4, 9, 320)).astype('float32'), 'y': np.array([0, 1, 0, 1]),
                'signal': rng.normal(size=(9, 800)).astype('float32'),
                'source_edf_sha256': f'{subject}-{run}', 'intervals': [[0., 2.5, 0], [2.5, 5., 1]]}
    monkeypatch.setattr(c.p, 'load_recording', record)
    c.run(plan, base, tmp_path / 'work', tmp_path / 'cache', 0, torch.device('cpu'))
    root = tmp_path / 'work/tiny_seed42'
    hashes = {str(f): c.p.digest(f) for f in root.rglob('epoch_*.pt')}
    assert len(hashes) == 6
    c.run(plan, base, tmp_path / 'work', tmp_path / 'cache', 0, torch.device('cpu'))
    assert hashes == {str(f): c.p.digest(f) for f in root.rglob('epoch_*.pt')}
    # Interrupted metrics write must recover from milestone checkpoint.
    missing = root / 'transfer/lr_0.001/S001/epoch_1.json'
    missing.unlink()
    c.run(plan, base, tmp_path / 'work', tmp_path / 'cache', 0, torch.device('cpu'))
    assert json.loads(missing.read_text())['target_r12_touched'] is False
    source = torch.load(root / 'source/best.pt', weights_only=True)['model']
    head = torch.load(root / 'head_only/lr_0.001/S001/epoch_2.pt', weights_only=True)['model']
    for key in source:
        if key.startswith('features.'):
            assert torch.allclose(source[key], head[key]), key
    altered = copy.deepcopy(plan)
    altered['target_lrs'] = [.002]
    with pytest.raises(ValueError, match='changed'):
        c.run(altered, base, tmp_path / 'work', tmp_path / 'cache', 0, torch.device('cpu'))
    assert not (root / 'RUNNING.lock').exists()


def test_epoch_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    args = dict(channels=9, samples=320, f1=2, depth=1, f2=2)
    c.p.seed_all(2)
    X, y = torch.randn(6, 9, 320), torch.tensor([0, 1, 0, 1, 0, 1])
    model = c.EEGNet(**args)
    opt = torch.optim.Adam(model.parameters())
    c.epoch(model, X, y, opt, 99, 3)
    c.p.save_checkpoint(tmp_path / 'resume.pt', dict(model=model.state_dict(), optimizer=opt.state_dict()))
    c.epoch(model, X, y, opt, 100, 3)
    restored = c.EEGNet(**args)
    opt2 = torch.optim.Adam(restored.parameters())
    state = torch.load(tmp_path / 'resume.pt', weights_only=True)
    restored.load_state_dict(state['model']); opt2.load_state_dict(state['optimizer'])
    c.epoch(restored, X, y, opt2, 100, 3)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key]), key
