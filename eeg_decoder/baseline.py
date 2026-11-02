import json
import platform
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import joblib
import mne
import numpy as np
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

LABELS = {"T0": 0, "T1": 1, "T2": 1}
CLASS_NAMES = ["REST", "HAND_IMAGERY"]
IMAGERY_RUNS = {4, 8, 12}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    subjects, runs = config["subjects"], config["runs"]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Choose unique subject IDs.")
    if any(type(s) is not int or not 1 <= s <= 109 for s in subjects):
        raise ValueError("PhysioNet subject IDs must be integers from 1 to 109.")
    if len(runs) < 2 or len(set(runs)) != len(runs) or not set(runs) <= IMAGERY_RUNS:
        raise ValueError("Use at least two of runs 4, 8, 12 (left/right fist imagery).")
    test_subjects = config["test_subjects"]
    if len(set(test_subjects)) != len(test_subjects) or not set(test_subjects) <= set(subjects):
        raise ValueError("test_subjects must contain unique IDs drawn from subjects.")
    if test_subjects and not set(subjects) - set(test_subjects):
        raise ValueError("Subject holdout must leave training subjects.")
    if config["test_run"] not in runs:
        raise ValueError("test_run must appear in runs.")
    channels = config["channels"]
    if len(set(channels)) != len(channels) or len(channels) < 3:
        raise ValueError("Select at least three unique channels.")
    if not 1 <= config["csp_components"] <= len(channels) - 1:
        raise ValueError("CSP components must fit the average-referenced channel rank.")
    low, high = config["bandpass_hz"]
    if not 0 < low < high < config["sample_rate_hz"] / 2:
        raise ValueError("Bandpass must fall between zero and the Nyquist frequency.")
    if config["window_seconds"] <= 0 or config["offset_seconds"] < 0:
        raise ValueError("Window must be positive and onset offset nonnegative.")
    if config["filter_order"] < 1 or not 0 <= config["csp_regularization"] <= 1:
        raise ValueError("Invalid filter order or CSP regularization.")


def preprocess_run(samples, config):
    """Process [selected channels, time] in volts; reset filter state per run."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] != len(config["channels"]):
        raise ValueError("Input channel count/order must match config.channels.")
    if not np.isfinite(samples).all():
        raise ValueError("EEG contains non-finite samples.")
    referenced = samples - samples.mean(axis=0, keepdims=True)
    sos = butter(
        config["filter_order"], config["bandpass_hz"],
        btype="bandpass", fs=config["sample_rate_hz"], output="sos",
    )
    return sosfilt(sos, referenced, axis=-1)


def extract_windows(samples, annotations, subject, run, config):
    """Use one equal-length window per annotation, entirely inside its state."""
    if run not in IMAGERY_RUNS:
        raise ValueError("T1/T2 mapping is only valid for runs 4, 8, 12 here.")
    sfreq = config["sample_rate_hz"]
    n_samples = int(round(config["window_seconds"] * sfreq))
    windows, metadata, skipped = [], [], []
    for index, annotation in enumerate(annotations):
        code = annotation["description"]
        if code not in LABELS:
            continue
        onset = float(annotation["onset"])
        duration = float(annotation["duration"])
        next_onset = (
            float(annotations[index + 1]["onset"])
            if index + 1 < len(annotations) else samples.shape[1] / sfreq
        )
        boundary = min(next_onset, onset + duration) if duration > 0 else next_onset
        start = int(round((onset + config["offset_seconds"]) * sfreq))
        stop = start + n_samples
        trial_id = f"S{subject:03d}R{run:02d}:{index}"
        if start < 0 or stop > samples.shape[1] or stop / sfreq > boundary + 1e-9:
            skipped.append({"trial_id": trial_id, "reason": "window_crosses_state_boundary"})
            continue
        windows.append(samples[:, start:stop])
        metadata.append({
            "trial_id": trial_id, "subject": subject, "run": run,
            "source_annotation": code, "label": CLASS_NAMES[LABELS[code]],
            "label_id": LABELS[code], "annotation_onset_seconds": onset,
            "window_start_seconds": start / sfreq, "window_end_seconds": stop / sfreq,
        })
    if not windows:
        raise ValueError(f"No valid windows for subject {subject}, run {run}.")
    return np.stack(windows), metadata, skipped


def prepare(config):
    validate_config(config)
    mne.set_log_level("WARNING")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    all_windows, records, skipped, sources = [], [], [], []
    jobs = [(subject, run) for subject in config["subjects"] for run in config["runs"]]

    def fetch(subject):
        files = eegbci.load_data(subject, config["runs"], path=str(data_dir), update_path=False)
        if len(files) != len(config["runs"]):
            raise ValueError("Expected one EDF per configured run.")
        return [((subject, run), [file]) for run, file in zip(config["runs"], files)]

    print(f"Fetching {len(jobs)} recordings (up to three downloads at a time).", flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        # One worker per subject avoids concurrent creation of the same cache directory.
        recording_files = dict(pair for group in executor.map(fetch, config["subjects"]) for pair in group)
    for subject in config["subjects"]:
        for run in config["runs"]:
            print(f"Preparing S{subject:03d}R{run:02d}", flush=True)
            files = recording_files[(subject, run)]
            if len(files) != 1:
                raise ValueError("Expected one EDF for each subject/run.")
            raw = mne.io.read_raw_edf(files[0], preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            if not np.isclose(raw.info["sfreq"], config["sample_rate_hz"]):
                raise ValueError(f"Unexpected sampling rate in {files[0]}; inspect before resampling.")
            raw.pick(config["channels"])
            raw.reorder_channels(config["channels"])
            samples = preprocess_run(raw.get_data(), config)
            annotations = [
                {"onset": onset, "duration": duration, "description": str(code)}
                for onset, duration, code in zip(
                    raw.annotations.onset, raw.annotations.duration, raw.annotations.description
                )
            ]
            windows, metadata, dropped = extract_windows(samples, annotations, subject, run, config)
            all_windows.append(windows)
            records.extend(metadata)
            skipped.extend(dropped)
            sources.append({"subject": subject, "run": run, "file": str(Path(files[0]).resolve())})
            raw.close()
    X = np.concatenate(all_windows)
    y = np.array([record["label_id"] for record in records], dtype=np.int64)
    np.savez_compressed(output / "windows.npz", X=X, y=y)
    write_json(output / "metadata.json", records)
    write_json(output / "prepared_config.json", config)
    audit = {
        "dataset": "PhysioNet EEGMMIDB 1.0.0", "sources": sources,
        "shape_trials_channels_samples": list(X.shape), "units": "volts",
        "class_counts": {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)},
        "annotation_mapping": {code: CLASS_NAMES[label] for code, label in LABELS.items()},
        "skipped_windows": skipped,
        "reference": "mean of the selected channels at each sample",
        "filter": "causal Butterworth SOS; zero initial state separately for each run",
        "artifact_handling": "finite-data validation only; no ICA or amplitude rejection yet",
    }
    write_json(output / "data_audit.json", audit)
    print(f"Prepared {X.shape}: {audit['class_counts']}. Audit: {output / 'data_audit.json'}")
    return audit


def split_indices(records, config):
    if config["test_subjects"]:
        is_test = np.array([record["subject"] in config["test_subjects"] for record in records])
        mode = "held_out_subjects"
    else:
        is_test = np.array([record["run"] == config["test_run"] for record in records])
        mode = "held_out_run_same_subjects"
    train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Both training and test partitions must contain windows.")
    train_runs = {(records[i]["subject"], records[i]["run"]) for i in train_idx}
    test_runs = {(records[i]["subject"], records[i]["run"]) for i in test_idx}
    if train_runs & test_runs:
        raise ValueError("The same recording appears in training and test.")
    return train_idx, test_idx, mode


def score(y, predicted):
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "n_windows": len(y), "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "rest_false_activation_fraction": float(matrix[0, 1] / matrix[0].sum()) if matrix[0].sum() else None,
        "report": classification_report(y, predicted, labels=[0, 1], target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0),
    }


def train(config):
    validate_config(config)
    output = Path(config["output_dir"])
    prepared = json.loads((output / "prepared_config.json").read_text(encoding="utf-8"))
    if prepared != config:
        raise ValueError("Config changed after prepare. Run prepare again, preferably in a new output_dir.")
    with np.load(output / "windows.npz", allow_pickle=False) as arrays:
        X, y = arrays["X"], arrays["y"]
    records = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if len(records) != len(y) or len(X) != len(y):
        raise ValueError("Windows and metadata are misaligned.")
    train_idx, test_idx, mode = split_indices(records, config)
    for name, indices in (("train", train_idx), ("test", test_idx)):
        if set(y[indices].tolist()) != {0, 1}:
            raise ValueError(f"{name} partition must contain both classes.")
    model = Pipeline([
        ("csp", CSP(n_components=config["csp_components"], reg=config["csp_regularization"],
                    log=True, norm_trace=False, rank={"eeg": len(config["channels"]) - 1})),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    mne.set_log_level("WARNING")
    model.fit(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx])
    probabilities = model.predict_proba(X[test_idx])
    dummy = DummyClassifier(strategy="most_frequent").fit(X[train_idx], y[train_idx])
    baseline = score(y[test_idx], dummy.predict(X[test_idx]))
    metrics = score(y[test_idx], predicted)
    per_subject = {}
    for subject in sorted({records[i]["subject"] for i in test_idx}):
        mask = np.array([records[i]["subject"] == subject for i in test_idx])
        per_subject[str(subject)] = score(y[test_idx][mask], predicted[mask])
    report = {
        "evaluation": mode, "class_order": CLASS_NAMES,
        "training_windows": len(train_idx), "test_windows": len(test_idx),
        "training_trial_ids": [records[i]["trial_id"] for i in train_idx],
        "test_trial_ids": [records[i]["trial_id"] for i in test_idx],
        "csp_lda": metrics, "majority_baseline": baseline, "per_subject": per_subject,
        "probabilities_are_calibrated": False,
        "versions": {name: version(name) for name in (
            "mne", "numpy", "scipy", "scikit-learn", "matplotlib", "joblib"
        )}, "python": platform.python_version(),
    }
    joblib.dump({"model": model, "config": config, "class_names": CLASS_NAMES}, output / "model.joblib")
    write_json(output / "metrics.json", report)
    predictions = []
    for row, index in enumerate(test_idx):
        predictions.append({
            **records[index], "predicted_intent": CLASS_NAMES[int(predicted[row])],
            "probabilities": {CLASS_NAMES[int(label)]: float(probabilities[row, col])
                              for col, label in enumerate(model.classes_)},
            "available_at_recording_seconds": records[index]["window_end_seconds"],
        })
    write_json(output / "test_predictions.json", predictions)
    plot_results(X[test_idx], y[test_idx], predicted, config, output)
    summary = (
        f"# First EEG baseline\n\nEvaluation: `{mode}`. "
        f"Train: {len(train_idx)} windows; test: {len(test_idx)} windows.\n\n"
        f"CSP + LDA balanced accuracy: {metrics['balanced_accuracy']:.3f}\n\n"
        f"Majority baseline balanced accuracy: {baseline['balanced_accuracy']:.3f}\n\n"
        f"CSP + LDA macro-F1: {metrics['macro_f1']:.3f}\n\n"
        f"Rest windows incorrectly predicted as imagery: {metrics['rest_false_activation_fraction']:.1%}\n\n"
        "| Subject | Balanced accuracy | Macro-F1 |\n|---|---:|---:|\n"
        + "".join(f"| {subject} | {scores['balanced_accuracy']:.3f} | {scores['macro_f1']:.3f} |\n"
                  for subject, scores in per_subject.items())
        + "\nThis is a small cue-based baseline. It does not establish continuous intent detection. "
        "Imagery labels describe instructed opening/closing, not separate grasp or wrist commands. "
        "Probabilities are uncalibrated. Keep the test partition fixed; use training-only validation "
        "before changing hyperparameters.\n"
    )
    (output / "results.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved model, plots, metrics and predictions in {output}")
    return report


def plot_results(X, y, predicted, config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(y, predicted, labels=[0, 1], display_labels=CLASS_NAMES,
                                           colorbar=False, ax=axes[0], cmap="Blues")
    axes[0].set_title("Held-out windows: CSP + LDA")
    channel = config["channels"].index("C3") if "C3" in config["channels"] else 0
    frequencies, psd = welch(X[:, channel], fs=config["sample_rate_hz"], nperseg=160, axis=-1)
    for label, name in enumerate(CLASS_NAMES):
        axes[1].plot(frequencies, 10 * np.log10(np.maximum(psd[y == label].mean(axis=0), 1e-30)), label=name)
    axes[1].set(xlim=(4, 40), xlabel="Frequency (Hz)", ylabel="Power (dB V^2/Hz)",
                title=f"{config['channels'][channel]}: mean test-window spectra")
    axes[1].legend()
    figure.savefig(output / "diagnostics.png", dpi=150)
    plt.close(figure)
