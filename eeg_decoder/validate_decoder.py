"""Select on development recordings, then evaluate a frozen fresh-person cohort.

The confirmation participants still provide runs 4/8 for individual calibration.
This does not measure zero-calibration transfer. Run once into a new directory.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from .baseline import load_config, prepare, write_json
from .compare_subjects import compare


def run(config, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    development = deepcopy(config)
    development.update(subjects=[1, 2, 3, 4, 5], runs=[4, 8], test_run=8,
                       test_subjects=[], output_dir=str(output / 'development'))
    write_json(output / 'protocol.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'development': 'S001-S005: fit run4; validation run8; do not use run12',
        'confirmation': 'S006-S010: individual calibration runs4/8, evaluation run12',
        'selection': 'highest individual balanced accuracy on development; CSP wins ties',
        'candidates': ['csp', 'riemannian'],
        'preprocessing': config,
        'versions': {p: version(p) for p in ('numpy', 'mne', 'scikit-learn', 'pyriemann')},
        'scope': 'Same-person calibrated decoding, not zero-calibration transfer or robot control',
    })
    prepare(development)
    scores = {}
    for family in ('csp', 'riemannian'):
        report = compare(development, output / f'validation_{family}', family)
        scores[family] = report['individual']['balanced_accuracy']
    chosen = max(scores, key=scores.get)
    # Save the choice before fetching or evaluating the confirmation recordings.
    write_json(output / 'frozen_selection.json', {
        'model_family': chosen, 'validation_scores': scores,
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'confirmation_subjects': [6, 7, 8, 9, 10], 'test_run': 12,
    })
    confirmation = deepcopy(config)
    confirmation.update(subjects=[6, 7, 8, 9, 10], runs=[4, 8, 12], test_run=12,
                        test_subjects=[], output_dir=str(output / 'confirmation'))
    prepare(confirmation)
    report = compare(confirmation, output / 'confirmation_models', chosen)
    report['limitation'] = ('Frozen model-family confirmation on S006-S010 run12, '
        'with same-person calibration runs4/8. Small cohort; probabilities uncalibrated; '
        'no robot outcome measured. Do not reuse as fresh confirmation after tuning.')
    write_json(output / 'confirmation_metrics.json', report)
    write_json(output / 'confirmation_models' / 'metrics.json', report)
    write_json(output / 'confirmation_config.json', confirmation)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.output)
