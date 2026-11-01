import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from eeg_decoder import compare_subjects, validate_decoder
from eeg_decoder.baseline import load_config
from fri_bci_sim import streaming


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs" / "baseline.json")

class SyntheticPipeline:
    """Tiny sklearn-like estimator driven by the label encoded in X."""

    def __init__(self, steps):
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return X[:, 0, 0].astype(int)

    def predict_proba(self, X):
        labels = self.predict(X)
        return np.column_stack((1 - labels, labels)).astype(float)


def _prepared_comparison(tmp_path, config):
    records, X, y = [], [], []
    for subject in (1, 2):
        for run in (4, 8, 12):
            for label in (0, 1):
                # Two windows per class in each run.
                for trial in range(2):
                    X.append(np.full((3, 4), float(label)))
                    y.append(label)
                    records.append({
                        "trial_id": f"S{subject:03d}R{run:02d}:{label * 2 + trial}",
                        "subject": subject, "run": run, "label_id": label,
                        "window_end_seconds": float(2 + trial),
                    })
    source = tmp_path / "prepared"
    source.mkdir()
    config = dict(config, subjects=[1, 2], runs=[4, 8, 12], test_run=12,
                  test_subjects=[], channels=["C3", "Cz", "C4"],
                  output_dir=str(source), csp_components=2,
                  csp_regularization=0.1)
    (source / "prepared_config.json").write_text(json.dumps(config))
    (source / "metadata.json").write_text(json.dumps(records))
    np.savez_compressed(source / "windows.npz", X=np.asarray(X), y=np.asarray(y))
    return config


def test_comparison_keeps_training_and_test_recordings_separate(tmp_path, config, monkeypatch):
    config = _prepared_comparison(tmp_path, config)
    monkeypatch.setattr(compare_subjects, "Pipeline", SyntheticPipeline)
    monkeypatch.setattr(compare_subjects, "CSP", lambda **kwargs: object())

    report = compare_subjects.compare(config, tmp_path / "comparison", "csp")

    assert report["evaluation"] == "held_out_run_same_subjects"
    for details in report["per_subject"].values():
        train_ids, test_ids = set(details["training_trial_ids"]), set(details["test_trial_ids"])
        assert train_ids.isdisjoint(test_ids)
        assert {trial_id.split("R")[1][:2] for trial_id in train_ids} == {"04", "08"}
        assert {trial_id.split("R")[1][:2] for trial_id in test_ids} == {"12"}


def test_comparison_predictions_match_saved_calibration_models(tmp_path, config, monkeypatch):
    config = _prepared_comparison(tmp_path, config)
    monkeypatch.setattr(compare_subjects, "Pipeline", SyntheticPipeline)
    monkeypatch.setattr(compare_subjects, "CSP", lambda **kwargs: object())
    destination = tmp_path / "comparison"

    compare_subjects.compare(config, destination, "csp")

    predictions = json.loads((destination / "test_predictions.json").read_text())
    by_subject = {}
    for row in predictions:
        by_subject.setdefault(row["subject"], []).append(row)
    for subject, rows in by_subject.items():
        artifact = joblib.load(destination / f"S{subject:03d}_model.joblib")
        assert artifact["calibration_subject"] == subject
        assert artifact["class_names"] == ["REST", "HAND_IMAGERY"]
        for row in rows:
            encoded_window = np.full((1, 3, 4), float(row["label_id"]))
            model_prediction = artifact["model"].predict(encoded_window)[0]
            assert row["predicted_intent"] == artifact["class_names"][model_prediction]
            assert set(row["probabilities"]) == set(artifact["class_names"])
            assert row["available_at_recording_seconds"] == row["window_end_seconds"]


def test_validation_development_partition_never_uses_run12(tmp_path, config, monkeypatch):
    events = []
    monkeypatch.setattr(validate_decoder, "version", lambda package: "test-version")
    monkeypatch.setattr(validate_decoder, "prepare", lambda cfg: events.append(("prepare", cfg.copy())))

    def fake_compare(cfg, destination, family):
        events.append(("compare", cfg.copy(), family))
        Path(destination).mkdir(parents=True)
        return {"individual": {"balanced_accuracy": 0.6 if family == "csp" else 0.5}}

    monkeypatch.setattr(validate_decoder, "compare", fake_compare)
    validate_decoder.run(config, tmp_path / "validation")

    development = next(event[1] for event in events if event[0] == "prepare")
    assert development["subjects"] == [1, 2, 3, 4, 5]
    assert development["runs"] == [4, 8]
    assert development["test_run"] == 8
    assert 12 not in development["runs"]
    assert all(event[1]["test_run"] == 8 for event in events if event[0] == "compare" and event[1]["subjects"] == [1, 2, 3, 4, 5])


def test_family_selection_is_frozen_before_confirmation_evaluation(tmp_path, config, monkeypatch):
    events = []
    real_write_json = validate_decoder.write_json

    def tracked_write_json(path, value):
        real_write_json(path, value)
        if Path(path).name == "frozen_selection.json":
            events.append(("frozen", value.copy()))
        else:
            events.append(("write", Path(path).name))

    monkeypatch.setattr(validate_decoder, "write_json", tracked_write_json)
    monkeypatch.setattr(validate_decoder, "version", lambda package: "test-version")

    def fake_prepare(cfg):
        events.append(("prepare", cfg.copy()))

    def fake_compare(cfg, destination, family):
        events.append(("compare", cfg.copy(), family))
        Path(destination).mkdir(parents=True)
        if cfg["subjects"] == [1, 2, 3, 4, 5]:
            score = {"csp": 0.7, "riemannian": 0.8}[family]
            return {"individual": {"balanced_accuracy": score}}
        return {"individual": {"balanced_accuracy": 0.5}, "pooled": {}}

    monkeypatch.setattr(validate_decoder, "prepare", fake_prepare)
    monkeypatch.setattr(validate_decoder, "compare", fake_compare)

    validate_decoder.run(config, tmp_path / "validation")

    frozen_index = next(i for i, event in enumerate(events) if event[0] == "frozen")
    confirmation_prepare_index = next(i for i, event in enumerate(events)
                                      if event[0] == "prepare" and event[1]["subjects"] == [6, 7, 8, 9, 10])
    confirmation_calls = [event for event in events if event[0] == "compare"
                          and event[1]["subjects"] == [6, 7, 8, 9, 10]]
    assert frozen_index < confirmation_prepare_index
    assert len(confirmation_calls) == 1
    assert confirmation_calls[0][2] == "riemannian"
    assert confirmation_calls[0][1]["test_run"] == 12
    assert confirmation_calls[0][1]["runs"] == [4, 8, 12]
    frozen = events[frozen_index][1]
    assert frozen["model_family"] == "riemannian"
    assert frozen["validation_scores"] == {"csp": 0.7, "riemannian": 0.8}




def test_person_specific_streaming_model_rejects_other_recording_subject(tmp_path, monkeypatch):
    artifact_path = tmp_path / "model.joblib"
    artifact_path.write_bytes(b"mock artifact")
    edf_path = tmp_path / "S007R12.edf"
    output_path = tmp_path / "predictions.jsonl"
    artifact = {
        "class_names": ["REST", "HAND_IMAGERY"],
        "config": {"sample_rate_hz": 160, "channels": ["C3", "Cz", "C4"]},
        "calibration_subject": 6,
    }
    monkeypatch.setattr(joblib, "load", lambda path: artifact)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Must reject subject mismatch before opening recording")

    import mne
    monkeypatch.setattr(mne.io, "read_raw_edf", unexpected_read)
    with pytest.raises(ValueError, match="calibration subject"):
        streaming.decode_recording(artifact_path, edf_path, output_path)
    assert not output_path.exists()
