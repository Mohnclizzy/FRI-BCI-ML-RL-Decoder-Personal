import json
from pathlib import Path
import sys

import pytest

from eeg_decoder import summarize_dense_ablation as report


EVIDENCE = Path(__file__).resolve().parents[1] / 'results/dense_window_ablation_2026-11-14'


def test_report_for_trainer_layout_uses_observed_metrics(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    for seed in (42, 123, 2026):
        folder = run / f'seed_{seed}'
        folder.mkdir(parents=True)
        rows = json.loads((EVIDENCE / f'seed_{seed}_target_summary.json').read_text())
        for row in rows:
            row['metrics']['window_metrics']['balanced_accuracy'] = 0.75
        (folder / 'target_summary.json').write_text(json.dumps(rows))
    reference = tmp_path / 'gpu_campaign_2026-11-13'
    reference.mkdir()
    baseline = {'complete': True, 'completed': 45, 'mean_balanced_accuracy': 0.625,
                'mean_raw_rest_activation': 0.4, 'mean_raw_missed_imagery': 0.5}
    (reference / 'development_summary.json').write_text(json.dumps({
        'groups': {'eegnet8/transfer/0.0001/300': baseline}}))
    monkeypatch.setattr(sys, 'argv', ['report', '--run-dir', str(run)])
    report.main()
    assert '| balanced accuracy | 62.50% | 75.00% |' in (run / 'summary.md').read_text()
    assert json.loads((run / 'aggregate.json').read_text())['n_seed_subject_rows'] == 45


def test_partial_seed_set_cannot_be_published(tmp_path):
    for seed in (42, 123):
        name = f'seed_{seed}_target_summary.json'
        (tmp_path / name).write_bytes((EVIDENCE / name).read_bytes())
    with pytest.raises(ValueError, match='42, 123 and 2026'):
        report.read_complete_cohort(tmp_path)
