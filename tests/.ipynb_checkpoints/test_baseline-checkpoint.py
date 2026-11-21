import json
from pathlib import Path

import numpy as np
import mne
import pytest

from eeg_decoder.baseline import extract_windows, load_config, prepare, preprocess_run, score, split_indices, train


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs" / "baseline.json")


def test_mapping_shape_and_state_boundaries(config):
    samples = np.zeros((9, 160 * 13))
    annotations = [
        {"onset": 0, "duration": 4, "description": "T0"},
        {"onset": 4, "duration": 4, "description": "T1"},
        {"onset": 8, "duration": 4, "description": "T2"},
        {"onset": 12, "duration": 1, "description": "T0"},
    ]
    X, records, skipped = extract_windows(samples, annotations, 1, 4, config)
    assert X.shape == (3, 9, 320)
    assert [record["label_id"] for record in records] == [0, 1, 1]
    assert len(skipped) == 1


def test_next_annotation_limits_overlong_duration(config):
    annotations = [
        {"onset": 0, "duration": 10, "description": "T0"},
        {"onset": 2, "duration": 4, "description": "T1"},
    ]
    _, records, skipped = extract_windows(np.zeros((9, 160 * 6)), annotations, 1, 4, config)
    assert len(records) == 1 and records[0]["source_annotation"] == "T1"
    assert len(skipped) == 1


def test_execution_and_foot_runs_are_rejected(config):
    with pytest.raises(ValueError, match="only valid"):
        extract_windows(np.zeros((9, 640)), [], 1, 6, config)


def test_causal_preprocessing_does_not_use_future(config):
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(9, 640)) * 1e-6
    changed = samples.copy()
    changed[:, 400:] *= 20
    original_result = preprocess_run(samples, config)
    changed_result = preprocess_run(changed, config)
    np.testing.assert_allclose(original_result[:, :400], changed_result[:, :400])
    np.testing.assert_allclose(original_result.mean(axis=0), 0, atol=1e-12)


def test_run_and_subject_holdouts(config):
    records = [{"subject": subject, "run": run} for subject in (1, 2) for run in (4, 8, 12)]
    train_idx, test_idx, mode = split_indices(records, config)
    assert test_idx.tolist() == [2, 5]
    assert mode == "held_out_run_same_subjects"
    assert set(train_idx) & set(test_idx) == set()
    config["test_subjects"] = [2]
    _, test_idx, mode = split_indices(records, config)
    assert test_idx.tolist() == [3, 4, 5]
    assert mode == "held_out_subjects"


def test_false_activations_are_separate_from_accuracy():
    result = score(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert result["balanced_accuracy"] == 0.75
    assert result["rest_false_activation_fraction"] == 0.5


def test_training_artifacts_on_synthetic_windows(tmp_path, config):
    config["output_dir"] = str(tmp_path)
    rng = np.random.default_rng(42)
    X, y, records = [], [], []
    for run in config["runs"]:
        for trial in range(20):
            label = trial % 2
            samples = rng.normal(size=(9, 320)) * 1e-6
            samples[label] *= 5
            X.append(samples - samples.mean(axis=0))
            y.append(label)
            records.append({"trial_id": f"1:{run}:{trial}", "subject": 1, "run": run,
                            "label_id": label, "window_end_seconds": float(trial * 4 + 3)})
    np.savez_compressed(tmp_path / "windows.npz", X=np.array(X), y=np.array(y))
    (tmp_path / "prepared_config.json").write_text(json.dumps(config))
    (tmp_path / "metadata.json").write_text(json.dumps(records))
    report = train(config)
    assert report["training_windows"] == 40
    assert report["test_windows"] == 20
    assert report["csp_lda"]["balanced_accuracy"] > 0.9
    predictions = json.loads((tmp_path / "test_predictions.json").read_text())
    assert len(predictions) == 20
    assert set(predictions[0]["probabilities"]) == {"REST", "HAND_IMAGERY"}
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "diagnostics.png").exists()


def test_changed_preprocessing_requires_reprepare(tmp_path, config):
    config["output_dir"] = str(tmp_path)
    (tmp_path / "prepared_config.json").write_text(json.dumps(config))
    config["bandpass_hz"] = [7, 30]
    with pytest.raises(ValueError, match="Config changed"):
        train(config)


def test_preparation_records_provenance_with_mocked_edf(tmp_path, config, monkeypatch):
    config["output_dir"] = str(tmp_path / "prepared")
    config["data_dir"] = str(tmp_path / "recordings")

    def fake_load(subject, runs, **kwargs):
        return [str(tmp_path / f"S{subject:03d}R{run:02d}.edf") for run in runs]

    def fake_read(*args, **kwargs):
        rng = np.random.default_rng(1)
        raw = mne.io.RawArray(rng.normal(size=(9, 1920)) * 1e-6,
                             mne.create_info(config["channels"], 160, "eeg"), verbose=False)
        raw.set_annotations(mne.Annotations([0, 4, 8], [4, 4, 4], ["T0", "T1", "T2"]))
        return raw

    monkeypatch.setattr("eeg_decoder.baseline.eegbci.load_data", fake_load)
    monkeypatch.setattr("eeg_decoder.baseline.mne.io.read_raw_edf", fake_read)
    audit = prepare(config)
    assert audit["shape_trials_channels_samples"] == [45, 9, 320]
    assert audit["class_counts"] == {"REST": 15, "HAND_IMAGERY": 30}
    records = json.loads((tmp_path / "prepared" / "metadata.json").read_text())
    assert records[0]["subject"] == 1 and records[-1]["subject"] == 5
    assert records[-1]["source_annotation"] == "T2"
    assert len({record["trial_id"] for record in records}) == 45
