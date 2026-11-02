"""Compare pooled and individual CSP/LDA on identical held-out recordings.

Run after prepare: python -m eeg_decoder.compare_subjects --config CONFIG
This is a diagnostic comparison, not hyperparameter selection or calibration.
"""
import argparse
import json
import hashlib
from importlib.metadata import version
from pathlib import Path

import joblib
import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from .baseline import CLASS_NAMES, load_config, score, split_indices, write_json


def compare(config, destination, family='csp'):
    if config['test_subjects']:
        raise ValueError('Individual calibration requires same-subject held-out runs.')
    source = Path(config['output_dir'])
    if json.loads((source / 'prepared_config.json').read_text()) != config:
        raise ValueError('Prepared configuration differs; prepare again in a new directory.')
    records = json.loads((source / 'metadata.json').read_text())
    with np.load(source / 'windows.npz', allow_pickle=False) as data:
        X, y = data['X'], data['y']
    if len(X) != len(y) or len(records) != len(y):
        raise ValueError('Windows and metadata are misaligned.')
    if any(r['label_id'] != int(label) for r, label in zip(records, y)):
        raise ValueError('Metadata labels differ from saved arrays.')
    train, test, mode = split_indices(records, config)
    subjects = np.array([r['subject'] for r in records])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    mne.set_log_level('WARNING')

    def fit(indices):
        if set(y[indices]) != {0, 1}:
            raise ValueError('Every training partition must contain both classes.')
        if family == 'riemannian':
            from pyriemann.estimation import Covariances
            from pyriemann.tangentspace import TangentSpace
            model = Pipeline([
                ('covariance', Covariances(estimator='oas')),
                ('tangent', TangentSpace(metric='riemann', tsupdate=False)),
                ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
            ])
            return model.fit(X[indices], y[indices])
        if family != 'csp':
            raise ValueError('Unknown model family')
        model = Pipeline([
            ('csp', CSP(n_components=config['csp_components'],
                        reg=config['csp_regularization'], log=True,
                        norm_trace=False, rank={'eeg': len(config['channels']) - 1})),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ])
        return model.fit(X[indices], y[indices])

    pooled = fit(train)
    pooled_prediction = pooled.predict(X[test])
    individual_prediction = np.empty(len(test), dtype=int)
    details, predictions = {}, []
    for subject in sorted(set(subjects[test])):
        training = train[subjects[train] == subject]
        mask = subjects[test] == subject
        testing = test[mask]
        if set(y[testing]) != {0, 1}:
            raise ValueError('Every evaluation subject must contain both classes.')
        model = fit(training)
        prediction = model.predict(X[testing])
        probabilities = model.predict_proba(X[testing])
        individual_prediction[mask] = prediction
        details[str(subject)] = {
            'pooled': score(y[testing], pooled_prediction[mask]),
            'individual': score(y[testing], prediction),
            'training_trial_ids': [records[i]['trial_id'] for i in training],
            'test_trial_ids': [records[i]['trial_id'] for i in testing],
        }
        joblib.dump({'model': model, 'config': config, 'class_names': CLASS_NAMES,
                     'calibration_subject': int(subject), 'model_family': family},
                    destination / f'S{subject:03d}_model.joblib')
        for row, index in enumerate(testing):
            predictions.append({**records[index],
                'predicted_intent': CLASS_NAMES[int(prediction[row])],
                'probabilities': {CLASS_NAMES[int(c)]: float(probabilities[row, j])
                                  for j, c in enumerate(model.classes_)},
                'available_at_recording_seconds': records[index]['window_end_seconds']})
    report = {'evaluation': mode, 'config': config, 'model_family': family,
              'prepared_windows_sha256': hashlib.sha256((source / 'windows.npz').read_bytes()).hexdigest(),
              'versions': {p: version(p) for p in ('numpy', 'mne', 'scikit-learn')},
              'pooled': score(y[test], pooled_prediction),
              'individual': score(y[test], individual_prediction),
              'per_subject': details, 'probabilities_are_calibrated': False,
              'limitation': 'Previously inspected test run; exploratory diagnostic, not independent confirmation. Individual models require calibration from that same person.'}
    write_json(destination / 'metrics.json', report)
    write_json(destination / 'test_predictions.json', predictions)
    print(json.dumps({k: report[k] for k in ('pooled', 'individual')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['csp', 'riemannian'], default='csp')
    args = parser.parse_args()
    compare(load_config(args.config), args.output, args.family)
