"""Tests for Phase 2 Stieger manifest, strict causality adapter, resumable trainer, and notebook."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eeg_decoder import phase2_train  # noqa: E402
from eeg_decoder.stieger import (  # noqa: E402
    CANONICAL_9_CHANNELS,
    STATE_CUE_ACTIVE_MI,
    STATE_CUE_DELIBERATE_RELAX,
    STATE_FEEDBACK_ACTIVE_MI,
    STATE_FEEDBACK_DELIBERATE_RELAX,
    STATE_INTERTRIAL_IDLE,
    STATE_POST_TRIAL_OUTCOME,
    STATE_UNLABELED_OR_ARTIFACT,
    extract_causal_windows,
    inspect_and_convert_mat,
    load_stieger_shard,
    sha256_file,
    validate_split_config,
)


def _make_synthetic_stieger_mat(
    path: Path,
    rng: np.random.Generator,
    perturb_trial0_after_idx: int | None = None,
) -> Path:
    n_ch = 62
    labels = [f"CH{i+1}" for i in range(n_ch)]
    # Place canonical channels using lowercase 'Cz' to test case-insensitive matching
    placed = [17, 19, 25, 26, 27, 28, 29, 35, 37]
    names = ["FC1", "FC2", "C3", "C1", "Cz", "C2", "C4", "CP1", "CP2"]
    for idx, nm in zip(placed, names):
        labels[idx] = "cz" if nm == "Cz" else nm

    targets = [1, 4, 2, 3]  # Right(MI), Down(Deliberate Relax), Left(MI), Up(MI)
    runs = [1, 1, 2, 2]
    artifacts = [0, 0, 0, 1]
    resinds = [7000, 6800, 7200, 6500]  # feedback 3.0s, 2.8s, 3.2s, 2.5s

    data_cells = np.empty((1, 4), dtype=object)
    time_cells = np.empty((1, 4), dtype=object)
    px_cells = np.empty((1, 4), dtype=object)
    py_cells = np.empty((1, 4), dtype=object)
    td_recs = []

    for k in range(4):
        n_t = resinds[k] + 1000
        tms = np.arange(-2000, -2000 + n_t, dtype=np.float64)
        # Smooth oscillatory EEG in 10-20 Hz so covariance & filters behave realistically
        t_s = np.arange(n_t, dtype=np.float64) / 1000.0
        sig = np.vstack([
            10.0 * np.sin(2.0 * np.pi * (10.0 + 0.3 * c) * t_s + 0.2 * c + k)
            + 2.0 * rng.standard_normal(n_t)
            for c in range(n_ch)
        ])
        if k == 0 and perturb_trial0_after_idx is not None:
            sig[:, perturb_trial0_after_idx:] += 500.0
        px = np.full(n_t, np.nan)
        py = np.full(n_t, np.nan)
        px[4000 : resinds[k]] = 0.5
        py[4000 : resinds[k]] = 0.5

        data_cells[0, k] = sig
        time_cells[0, k] = tms.reshape(1, -1)
        px_cells[0, k] = px.reshape(1, -1)
        py_cells[0, k] = py.reshape(1, -1)
        td_recs.append((
            np.array([[1 if k < 2 else 2]], dtype=np.int16),
            np.array([[runs[k]]], dtype=np.int16),
            np.array([[k + 1]], dtype=np.int16),
            np.array([[targets[k]]], dtype=np.int16),
            np.array([[(resinds[k] - 4000) / 1000.0]]),
            np.array([[targets[k]]], dtype=np.float64),
            np.array([[resinds[k]]], dtype=np.int32),
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[artifacts[k]]], dtype=np.uint8),
        ))

    td_dtype = np.dtype([
        ("tasknumber", "O"),
        ("runnumber", "O"),
        ("trialnumber", "O"),
        ("targetnumber", "O"),
        ("triallength", "O"),
        ("targethitnumber", "O"),
        ("resultind", "O"),
        ("result", "O"),
        ("forcedresult", "O"),
        ("artifact", "O"),
    ])
    td_arr = np.array([td_recs], dtype=td_dtype)
    label_arr = np.empty((len(labels), 1), dtype=object)
    for idx, lb in enumerate(labels):
        label_arr[idx, 0] = lb
    chaninfo = {"label": label_arr, "noisechan": np.array([["6"]], dtype=object)}
    meta = {"age": 30, "gender": "F", "handedness": "R", "MBSRsubject": 1, "date": "20200101", "time": 10}
    bci = {
        "data": data_cells,
        "time": time_cells,
        "positionx": px_cells,
        "positiony": py_cells,
        "SRATE": np.array([[1000.0]]),
        "TrialData": td_arr,
        "metadata": meta,
        "chaninfo": chaninfo,
    }
    savemat(str(path), {"BCI": bci})
    return path


def _make_synthetic_shard(path: Path, subject_id: int, session_id: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    n_trials = 20
    srate = 100
    n_ch = 9
    chunks_eeg, chunks_t, chunks_s, offsets = [], [], [], [0]
    resind_100, targets, tasks, runs = [], [], [], []
    for k in range(n_trials):
        n_t = 900  # 9.0s: 2s idle (-2000..-10), 2s cue (0..1990), 4s fb (2000..5990), 1s post (6000..6990)
        t_ms = np.arange(-1990, -1990 + n_t * 10, 10, dtype=np.float32)
        t_s = np.arange(n_t, dtype=np.float64) / srate
        tgt = [1, 2, 3, 4][k % 4]
        freq = 12.0 if tgt in (1, 2, 3) else 10.0
        sig = np.vstack([
            (3e-6 if tgt in (1, 2, 3) else 1.5e-6) * np.sin(2 * np.pi * freq * t_s + 0.3 * c + 0.1 * k)
            + 0.5e-6 * rng.standard_normal(n_t)
            for c in range(n_ch)
        ]).astype(np.float32)
        st = np.full(n_t, STATE_POST_TRIAL_OUTCOME, dtype=np.int8)
        st[t_ms < 0] = STATE_INTERTRIAL_IDLE
        is_relax = tgt == 4
        st[(t_ms >= 0) & (t_ms < 2000)] = STATE_CUE_DELIBERATE_RELAX if is_relax else STATE_CUE_ACTIVE_MI
        st[(t_ms >= 2000) & (np.arange(n_t) < 800)] = (
            STATE_FEEDBACK_DELIBERATE_RELAX if is_relax else STATE_FEEDBACK_ACTIVE_MI
        )
        chunks_eeg.append(sig)
        chunks_t.append(t_ms)
        chunks_s.append(st)
        offsets.append(offsets[-1] + n_t)
        resind_100.append(799)
        targets.append(tgt)
        tasks.append(1 if tgt in (1, 2) else 2)
        runs.append(1 if k < 10 else 2)

    np.savez(
        path,
        eeg=np.concatenate(chunks_eeg, axis=1),
        times_ms=np.concatenate(chunks_t, axis=0),
        state_id=np.concatenate(chunks_s, axis=0),
        trial_offsets=np.asarray(offsets, dtype=np.int32),
        channels=np.asarray(CANONICAL_9_CHANNELS),
        subject_id=np.int32(subject_id),
        session_id=np.int32(session_id),
        srate_hz=np.int32(srate),
        tasknumber=np.asarray(tasks, dtype=np.int16),
        runnumber=np.asarray(runs, dtype=np.int16),
        trialnumber=np.arange(1, n_trials + 1, dtype=np.int32),
        targetnumber=np.asarray(targets, dtype=np.int16),
        triallength_s=np.full(n_trials, 4.0, dtype=np.float32),
        resultind_100hz=np.asarray(resind_100, dtype=np.int32),
        result=np.ones(n_trials, dtype=np.float32),
        forcedresult=np.ones(n_trials, dtype=np.float32),
        artifact=np.zeros(n_trials, dtype=bool),
        source_md5=np.asarray("synthetic"),
        source_sha256=np.asarray("synthetic"),
    )
    return path


def test_stieger_split_and_training_configs_invariants():
    split_cfg = json.loads((REPO_ROOT / "configs" / "phase2_stieger_split.json").read_text(encoding="utf-8"))
    validate_split_config(split_cfg)
    assert len(split_cfg["dev_subjects"]) == 8
    assert len(split_cfg["test_subjects"]) == 10
    assert len(split_cfg["source_train_subjects_order"]) == 44
    assert len(split_cfg["source_scales"]["scale_10pct"]) == 4
    assert len(split_cfg["source_scales"]["scale_30pct"]) == 13
    assert len(split_cfg["source_scales"]["scale_100pct"]) == 44

    train_cfg = json.loads((REPO_ROOT / "configs" / "phase2_training.json").read_text(encoding="utf-8"))
    assert train_cfg["preprocessing"]["channels"] == CANONICAL_9_CHANNELS
    assert train_cfg["preprocessing"]["primary_window_s"] == 2.0
    assert set(train_cfg["source_pretraining"]["classes"].keys()) == {"0", "1", "2"}
    assert train_cfg["source_pretraining"]["scales"] == ["scale_10pct", "scale_30pct", "scale_100pct"]
    assert train_cfg["source_pretraining"]["seeds"] == [0, 1, 2]
    assert "BCICIV-1" in train_cfg["target_calibration_and_evaluation"]["evaluation_endpoints"]["historical_dev_reference"]


def test_audit_artifacts_and_sha256_manifest():
    audit_dir = REPO_ROOT / "results" / "phase2_stieger_audit_2026-11-19"
    sums_path = audit_dir / "SHA256SUMS.json"
    assert sums_path.is_file()
    sums = json.loads(sums_path.read_text(encoding="utf-8"))
    expected_files = {"stieger_figshare_manifest.json", "audit_metrics.json", "summary.md", "cpu_smoke_test.json"}
    assert expected_files <= set(sums["files"].keys())
    for rel, expected_sha in sums["files"].items():
        fp = audit_dir / rel
        assert fp.is_file(), f"Missing audited artifact {fp}"
        assert sha256_file(fp) == expected_sha

    audit = json.loads((audit_dir / "audit_metrics.json").read_text(encoding="utf-8"))
    assert audit["audited_session_keys"] == ["S1_Session_1", "S1_Session_7", "S2_Session_1"]
    assert audit["corpus_summary"]["n_subjects"] == 62
    assert audit["corpus_summary"]["n_sessions"] == 598
    for sess in audit["sessions"]:
        assert sess["non_finite_canonical_samples"] == 0
        assert sess["state_minutes"]["feedback_deliberate_relax"] > 0.0
        assert sess["state_minutes"]["feedback_active_mi"] > 0.0
        assert sess["state_minutes"]["intertrial_idle"] > 0.0


def test_synthetic_stieger_mat_roundtrip_and_strict_causality(tmp_path):
    rng = np.random.default_rng(12345)
    t_cut_1000hz = 6200  # corresponds to t_ms = 4200 ms inside feedback of trial 0
    mat_orig = _make_synthetic_stieger_mat(tmp_path / "orig.mat", rng)
    rng_same = np.random.default_rng(12345)
    mat_pert = _make_synthetic_stieger_mat(tmp_path / "pert.mat", rng_same, perturb_trial0_after_idx=t_cut_1000hz)

    rep_o = inspect_and_convert_mat(mat_orig, tmp_path / "orig.npz", subject_id=99, session_id=1)
    rep_p = inspect_and_convert_mat(mat_pert, tmp_path / "pert.npz", subject_id=99, session_id=1)
    assert rep_o["noisechan_resolved"] == ["CH6"]
    assert rep_o["noisy_canonical_channels"] == []
    assert rep_o["artifact_trials"] == 1

    sh_o = load_stieger_shard(tmp_path / "orig.npz")
    sh_p = load_stieger_shard(tmp_path / "pert.npz")
    states = set(sh_o["state_id"].tolist())
    assert {
        STATE_INTERTRIAL_IDLE,
        STATE_CUE_ACTIVE_MI,
        STATE_CUE_DELIBERATE_RELAX,
        STATE_FEEDBACK_ACTIVE_MI,
        STATE_FEEDBACK_DELIBERATE_RELAX,
        STATE_POST_TRIAL_OUTCOME,
        STATE_UNLABELED_OR_ARTIFACT,
    } == states

    # Strictly verify future samples beyond 100 Hz cutoff in trial 0 cause ZERO change <= cutoff
    s0, e0 = int(sh_o["trial_offsets"][0]), int(sh_o["trial_offsets"][1])
    t0 = sh_o["times_ms"][s0:e0]
    cutoff_ms = float(sh_o["times_ms"][s0 + (t_cut_1000hz - 10) // 10 - 1])
    le_mask = np.zeros(sh_o["eeg"].shape[1], dtype=bool)
    le_mask[s0:e0] = t0 <= cutoff_ms
    assert np.max(np.abs(sh_o["eeg"][:, le_mask] - sh_p["eeg"][:, le_mask])) == 0.0

    # Windowed extraction also preserves exact 0.0 causality for all windows ending <= cutoff_ms
    w_o = extract_causal_windows(sh_o, window_s=1.0, step_s=0.25, use_step_zi=True)
    w_p = extract_causal_windows(sh_p, window_s=1.0, step_s=0.25, use_step_zi=True)
    w_mask = (w_o["trial_idx"] == 0) & (w_o["times_ms"] <= cutoff_ms)
    assert np.any(w_mask)
    assert np.max(np.abs(w_o["X"][w_mask] - w_p["X"][w_mask])) == 0.0


def test_phase2_train_checkpoint_verify_and_resume_equivalence(tmp_path):
    sh_dir = tmp_path / "shards"
    work_dir = tmp_path / "work"
    drive_dir = tmp_path / "drive"
    sh_dir.mkdir()
    tr1 = _make_synthetic_shard(sh_dir / "S90_Session_1.npz", 90, 1, seed=10)
    tr2 = _make_synthetic_shard(sh_dir / "S91_Session_1.npz", 91, 1, seed=20)
    va1 = _make_synthetic_shard(sh_dir / "S92_Session_1.npz", 92, 1, seed=30)

    # First run: stop cleanly after 2 updates
    rc1 = phase2_train.main([
        "train",
        "--shard-dir", str(sh_dir),
        "--work-dir", str(work_dir),
        "--drive-dir", str(drive_dir),
        "--train-shards", str(tr1), str(tr2),
        "--val-shards", str(va1),
        "--job-id", "unit_test_job",
        "--seed", "0",
        "--window-s", "1.0",
        "--max-updates", "4",
        "--stop-after-updates", "2",
        "--checkpoint-every", "2",
        "--eval-every", "2",
        "--batch-size", "16",
        "--device", "cpu",
    ])
    assert rc1 == 0
    st1 = json.loads((drive_dir / "unit_test_job" / "job_state.json").read_text(encoding="utf-8"))
    assert st1["step"] == 2
    assert st1["status"] == "in_progress"
    assert st1["drive_verified"] is True
    assert sha256_file(drive_dir / "unit_test_job" / "ckpt_latest.pt") == st1["ckpt_latest_sha256"]

    # Second run: resume in fresh state to step 4
    rc2 = phase2_train.main([
        "train",
        "--shard-dir", str(sh_dir),
        "--work-dir", str(work_dir),
        "--drive-dir", str(drive_dir),
        "--train-shards", str(tr1), str(tr2),
        "--val-shards", str(va1),
        "--job-id", "unit_test_job",
        "--seed", "0",
        "--window-s", "1.0",
        "--max-updates", "4",
        "--checkpoint-every", "2",
        "--eval-every", "2",
        "--batch-size", "16",
        "--device", "cpu",
        "--resume",
    ])
    assert rc2 == 0
    st2 = json.loads((drive_dir / "unit_test_job" / "job_state.json").read_text(encoding="utf-8"))
    assert st2["step"] == 4
    assert st2["status"] == "completed"
    assert st2["drive_verified"] is True
    assert sha256_file(drive_dir / "unit_test_job" / "ckpt_latest.pt") == st2["ckpt_latest_sha256"]
    assert np.isfinite(st2["val_metrics"]["val_loss"])


def test_phase2_notebook_valid_json_ast_and_pinned_hashes():
    nb_path = REPO_ROOT / "notebooks" / "FRI_BCI_Phase2_EEG_Scaling.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    cells = nb["cells"]
    assert len(cells) >= 6
    code_sources = []
    for idx, cell in enumerate(cells):
        src = "".join(cell["source"])
        if cell["cell_type"] == "code":
            ast.parse(src, filename=f"{nb_path.name}:cell_{idx}")
            code_sources.append(src)
    joined_code = "\n".join(code_sources)
    assert "RUN_CAMPAIGN = False" in joined_code
    for rel in (
        "eeg_decoder/neural.py",
        "eeg_decoder/stieger.py",
        "eeg_decoder/phase2_train.py",
        "configs/phase2_stieger_split.json",
        "configs/phase2_training.json",
    ):
        norm_sha = hashlib.sha256((REPO_ROOT / rel).read_text(encoding="utf-8").encode()).hexdigest()
        assert norm_sha in joined_code
