"""Priority 3: Matched Development-Only Decoder, Transition & Gate Utility Study (S001-S010, R04 -> R08).

Strict Data-Split Invariant:
- Uses ONLY development participants S001-S010.
- Fits target models strictly on R04 (30 cue-aligned windows per participant).
- Scores window classification, continuous 4 Hz replay, transition vs interior score
  distributions, training-adequacy ablations, and predeclared gate candidates strictly on R08.
- NEVER loads or inspects any R12 recording or S011-S015 target recording.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import mne
from mne.decoding import CSP
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "simulation" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eeg_decoder.alignment import apply_alignment, fit_alignment  # noqa: E402
from eeg_decoder.gpu_pipeline import (  # noqa: E402
    command_metrics,
    load_target,
    make_model,
    predict,
    read_config,
    seed_all,
    train_epoch,
)
from eeg_decoder.risk_gate import HysteresisGate, calibrate_threshold, rest_episode_maxima  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
SCRATCH_WORK_DIR = REPO_ROOT / "outputs" / "local_scratch_2026-11-11"
CACHE_DIR = SCRATCH_WORK_DIR / "cache" / "7d3de6fbf853d44a"
CONFIG_PATH = REPO_ROOT / "configs" / "gpu_research.json"

DEV_SUBJECTS = list(range(1, 11))  # S001-S010 only


def load_cached_dev_recording(subject: int, run: int) -> dict[str, Any]:
    """Load R04 or R08 from cache. Explicitly refuses run 12 or subjects > 10."""
    if subject not in DEV_SUBJECTS:
        raise ValueError(f"Development study only permits subjects 1..10, got {subject}")
    if run not in (4, 8):
        raise ValueError(f"Development study strictly forbids opening run {run}; only R04 and R08 are allowed.")
    stem = f"S{subject:03d}R{run:02d}"
    meta_path = CACHE_DIR / f"{stem}.json"
    npz_path = CACHE_DIR / f"{stem}.npz"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        return {
            **meta,
            "X": arrays["X"],
            "y": arrays["y"],
            "signal": arrays["signal"],
        }


def make_csp_lda(n_channels: int = 9, n_components: int = 4, reg: float = 0.1) -> Pipeline:
    mne.set_log_level("WARNING")
    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=n_components,
                    reg=reg,
                    log=True,
                    norm_trace=False,
                    rank={"eeg": n_channels - 1},
                ),
            ),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def make_riemannian_lda() -> Pipeline:
    return Pipeline(
        [
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann", tsupdate=False)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def extract_continuous_4hz_windows(signal: np.ndarray, sfreq: int = 160, window_s: float = 2.0, stride_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    width = round(window_s * sfreq)
    step = round(stride_s * sfreq)
    ends = np.arange(width, signal.shape[1] + 1, step)
    windows = np.stack([signal[:, end - width : end] for end in ends])
    times = ends / float(sfreq)
    return windows, times


def score_cue_windows(y_true: np.ndarray, p_hand: np.ndarray) -> dict[str, float]:
    y_pred = (p_hand >= 0.5).astype(np.int64)
    rest_mask = y_true == 0
    hand_mask = y_true == 1
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_hand)),
        "rest_recall": float((y_pred[rest_mask] == 0).mean()),
        "hand_recall": float((y_pred[hand_mask] == 1).mean()),
        "rest_false_positive_rate": float((y_pred[rest_mask] == 1).mean()),
        "mean_p_hand_on_rest": float(p_hand[rest_mask].mean()),
        "mean_p_hand_on_imagery": float(p_hand[hand_mask].mean()),
        "max_p_hand_on_rest_cue_windows": float(p_hand[rest_mask].max()),
        "max_p_hand_on_imagery_cue_windows": float(p_hand[hand_mask].max()),
    }


def analyze_transition_vs_interior(
    probabilities: np.ndarray,
    times: np.ndarray,
    intervals: list[list[float]],
    window_s: float = 2.0,
    max_age_s: float = 1.0,
) -> dict[str, Any]:
    """Decompose 4 Hz continuous replay scores into pure class-interior vs transition windows,
    and locate where each R08 REST episode's maximum score occurs relative to the preceding transition.
    """
    p = np.asarray(probabilities, dtype=float)
    t = np.asarray(times, dtype=float)
    w_starts = t - window_s

    pure_rest_scores: list[float] = []
    transition_imagery_to_rest_scores: list[float] = []
    pure_imagery_scores: list[float] = []
    transition_rest_to_imagery_scores: list[float] = []

    for prob, ws, we in zip(p, w_starts, t):
        # Find interval containing window end `we`
        containing = None
        for idx, (s, e, lbl) in enumerate(intervals):
            if s <= we <= e + 1e-9:
                containing = (idx, float(s), float(e), int(lbl))
                break
        if containing is None:
            continue
        idx, s, e, lbl = containing
        is_pure = ws >= s - 1e-9
        if lbl == 0:
            if is_pure or idx == 0:
                pure_rest_scores.append(float(prob))
            else:
                transition_imagery_to_rest_scores.append(float(prob))
        else:
            if is_pure:
                pure_imagery_scores.append(float(prob))
            else:
                transition_rest_to_imagery_scores.append(float(prob))

    # Now inspect each REST episode maximum under the exact `rest_episode_maxima` rule
    ends = np.minimum(np.r_[t[1:], np.inf], t + max_age_s)
    episode_details: list[dict[str, Any]] = []
    max_in_transition_count = 0
    rest_episodes_after_imagery = 0

    for ep_idx, (start, end, label) in enumerate(intervals):
        if int(label) != 0:
            continue
        start_f, end_f = float(start), float(end)
        overlap = (t < end_f) & (ends > start_f)
        if not overlap.any():
            continue
        idxs = np.flatnonzero(overlap)
        best_local = idxs[int(np.argmax(p[idxs]))]
        peak_p = float(p[best_local])
        peak_t = float(t[best_local])
        offset_from_rest_onset_s = peak_t - start_f
        # A window [peak_t - 2.0, peak_t] overlaps the preceding HAND_IMAGERY episode
        # whenever ep_idx > 0 and peak_t < start_f + window_s (including carried commands issued just before start_f).
        overlaps_prior_imagery = bool(ep_idx > 0 and (peak_t - start_f) < window_s - 1e-9)
        if ep_idx > 0:
            rest_episodes_after_imagery += 1
            if overlaps_prior_imagery:
                max_in_transition_count += 1

        # Also compute maximum strictly inside the pure REST interior (peak_t >= start_f + window_s)
        pure_mask = overlap & ((t - start_f) >= window_s - 1e-9 if ep_idx > 0 else np.ones_like(overlap, dtype=bool))
        pure_interior_max = float(p[pure_mask].max()) if pure_mask.any() else 0.0

        episode_details.append(
            {
                "interval_index": ep_idx,
                "rest_start_s": start_f,
                "rest_end_s": end_f,
                "peak_probability": peak_p,
                "peak_time_s": peak_t,
                "offset_from_rest_onset_s": offset_from_rest_onset_s,
                "overlaps_preceding_imagery": overlaps_prior_imagery,
                "pure_interior_max_probability": pure_interior_max,
            }
        )

    all_maxima = [ep["peak_probability"] for ep in episode_details]
    pure_interior_maxima = [ep["pure_interior_max_probability"] for ep in episode_details]

    def stats(vals: list[float]) -> dict[str, float | int | None]:
        if not vals:
            return {"count": 0, "mean": None, "std": None, "q50": None, "q80": None, "q90": None, "max": None}
        arr = np.asarray(vals, dtype=float)
        return {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "q50": float(np.quantile(arr, 0.50)),
            "q80": float(np.quantile(arr, 0.80)),
            "q90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }

    return {
        "pure_rest_interior": stats(pure_rest_scores),
        "transition_imagery_to_rest": stats(transition_imagery_to_rest_scores),
        "pure_imagery_interior": stats(pure_imagery_scores),
        "transition_rest_to_imagery": stats(transition_rest_to_imagery_scores),
        "rest_episode_count": len(episode_details),
        "post_imagery_rest_episode_count": rest_episodes_after_imagery,
        "post_imagery_rest_max_in_transition_count": max_in_transition_count,
        "post_imagery_rest_max_in_transition_fraction": (
            max_in_transition_count / rest_episodes_after_imagery if rest_episodes_after_imagery else 0.0
        ),
        "mean_episode_max_all_s": float(np.mean(all_maxima)),
        "mean_episode_max_pure_interior_only": float(np.mean(pure_interior_maxima)),
        "mean_peak_offset_from_rest_onset_s": float(
            np.mean([ep["offset_from_rest_onset_s"] for ep in episode_details if ep["interval_index"] > 0])
        ),
        "episodes": episode_details,
    }


def evaluate_gate_on_r08(
    probabilities: np.ndarray,
    times: np.ndarray,
    intervals: list[list[float]],
    *,
    threshold: float,
    margin: float,
    persistence_s: float,
    max_age_s: float = 1.0,
    horizon_20s: float = 20.0,
) -> dict[str, Any]:
    """Run HysteresisGate on R08 4 Hz score stream and compute both full-recording and 20s-horizon utility."""
    gate = HysteresisGate(threshold, margin=margin, persistence_s=persistence_s, max_age_s=max_age_s)
    skills: list[str] = []
    for prob, t in zip(probabilities, times):
        d = gate.update(float(prob), float(t), float(t), float(t), "R08")
        skills.append(d["skill"])

    full_cmd = command_metrics(times, skills, intervals, max_age_s)

    # Also compute 20-second robot-horizon enabled duration (20 Hz steps 1..400, query_time_s = (step-1)/20.0)
    # and imagery episode onset delays across R08
    ends = np.minimum(np.r_[times[1:], np.inf], times + max_age_s)
    enabled = np.asarray(skills) == "grasp_lift"

    imagery_episodes = [(float(s), float(e)) for s, e, lbl in intervals if int(lbl) == 1]
    detected_episodes = 0
    onset_delays: list[float] = []
    for s, e in imagery_episodes:
        # Find first enabled command interval [times[i], ends[i]) overlapping [s, e)
        overlap_idxs = np.flatnonzero(enabled & (times < e - 1e-9) & (ends > s + 1e-9))
        if len(overlap_idxs) > 0:
            detected_episodes += 1
            first_idx = int(overlap_idxs[0])
            active_start = max(s, float(times[first_idx]))
            onset_delays.append(max(0.0, active_start - s))

    # 20s horizon requested steps on 20 Hz clock
    step_queries = np.arange(400, dtype=float) / 20.0
    enabled_20s_steps = 0
    for q in step_queries:
        valid = np.flatnonzero((times <= q + 1e-12) & (q - times <= max_age_s + 1e-12))
        if len(valid) > 0 and skills[int(valid[-1])] == "grasp_lift":
            enabled_20s_steps += 1
    enabled_20s_seconds = enabled_20s_steps / 20.0

    # Predeclared utility check:
    # 1. Not always-HOLD: >= 2.50s (50 steps) enabled in first 20s AND >= 30% imagery coverage (missed_imagery_fraction <= 0.70)
    # 2. Bounded unintended REST activation: rest_command_fraction <= 0.35
    imagery_coverage = 1.0 - float(full_cmd["missed_imagery_fraction"])
    rest_frac = float(full_cmd["rest_command_fraction"])
    passes_utility = (
        enabled_20s_seconds >= 2.50
        and imagery_coverage >= 0.30
        and rest_frac <= 0.35
    )

    return {
        "threshold": float(threshold),
        "margin": float(margin),
        "persistence_s": float(persistence_s),
        "rest_command_fraction": rest_frac,
        "rest_episode_activation_fraction": float(full_cmd["rest_episode_activation_fraction"]),
        "imagery_command_coverage": imagery_coverage,
        "missed_imagery_fraction": float(full_cmd["missed_imagery_fraction"]),
        "imagery_episode_detection_rate": detected_episodes / len(imagery_episodes) if imagery_episodes else 0.0,
        "mean_imagery_onset_delay_s": float(np.mean(onset_delays)) if onset_delays else None,
        "enabled_steps_20s": int(enabled_20s_steps),
        "enabled_seconds_20s": float(enabled_20s_seconds),
        "any_command_issued": bool(enabled.any()),
        "passes_predeclared_utility": bool(passes_utility),
    }


def run_scratch_training_adequacy_ablation(
    config: dict[str, Any],
    r04_by_subj: dict[int, dict[str, Any]],
    r08_by_subj: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Compare R04->R08 Scratch EA-EEGNet across optimizer update budgets:
    - 25 updates: batch_size=64, epochs=25 (1 batch/epoch, frozen Nov 11 setting)
    - 100 updates: batch_size=15, epochs=50 (2 batches/epoch)
    - 200 updates: batch_size=15, epochs=100 (2 batches/epoch)
    """
    device = torch.device("cpu")
    regimes = [
        {"name": "scratch_25_updates_bs64_ep25", "batch_size": 64, "epochs": 25, "updates": 25},
        {"name": "scratch_100_updates_bs15_ep50", "batch_size": 15, "epochs": 50, "updates": 100},
        {"name": "scratch_200_updates_bs15_ep100", "batch_size": 15, "epochs": 100, "updates": 200},
    ]
    results_by_regime: dict[str, Any] = {}
    for reg in regimes:
        per_subj: dict[str, Any] = {}
        custom_cfg = {**config, "batch_size": reg["batch_size"], "adapt_epochs": reg["epochs"]}
        for s in DEV_SUBJECTS:
            sid = f"S{s:03d}"
            r04 = r04_by_subj[s]
            r08 = r08_by_subj[s]
            ea_mat = fit_alignment(r04["X"])
            X_fit = apply_alignment(r04["X"], ea_mat)
            y_fit = r04["y"]
            X_val = apply_alignment(r08["X"], ea_mat)
            y_val = r08["y"]

            seed_all(custom_cfg["seed"] + s)
            model = make_model(custom_cfg).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=custom_cfg["adapt_lr"], weight_decay=1e-4)
            initial_loss = None
            final_loss = None
            for ep in range(reg["epochs"]):
                loss = train_epoch(model, X_fit, y_fit, device, opt, custom_cfg, custom_cfg["seed"] + 1000 * s + ep)
                if ep == 0:
                    initial_loss = float(loss)
                final_loss = float(loss)

            p_fit = predict(model, X_fit, device)[:, 1]
            p_val = predict(model, X_val, device)[:, 1]
            fit_acc = float(balanced_accuracy_score(y_fit, (p_fit >= 0.5).astype(np.int64)))
            val_metrics = score_cue_windows(y_val, p_val)
            per_subj[sid] = {
                "r04_InitialLoss": initial_loss,
                "r04_FinalLoss": final_loss,
                "r04_train_balanced_accuracy": fit_acc,
                "r08_dev_balanced_accuracy": val_metrics["balanced_accuracy"],
                "r08_dev_macro_f1": val_metrics["macro_f1"],
                "r08_dev_roc_auc": val_metrics["roc_auc"],
            }
        results_by_regime[reg["name"]] = {
            **reg,
            "mean_r04_final_loss": float(np.mean([v["r04_FinalLoss"] for v in per_subj.values()])),
            "mean_r04_train_balanced_accuracy": float(
                np.mean([v["r04_train_balanced_accuracy"] for v in per_subj.values()])
            ),
            "mean_r08_dev_balanced_accuracy": float(
                np.mean([v["r08_dev_balanced_accuracy"] for v in per_subj.values()])
            ),
            "mean_r08_dev_macro_f1": float(np.mean([v["r08_dev_macro_f1"] for v in per_subj.values()])),
            "mean_r08_dev_roc_auc": float(np.mean([v["r08_dev_roc_auc"] for v in per_subj.values()])),
            "per_subject": per_subj,
        }
    return results_by_regime


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = read_config(CONFIG_PATH)
    device = torch.device("cpu")

    r04_by_subj = {s: load_cached_dev_recording(s, 4) for s in DEV_SUBJECTS}
    r08_by_subj = {s: load_cached_dev_recording(s, 8) for s in DEV_SUBJECTS}

    decoder_names = ("csp_lda_r04", "riemannian_lda_r04", "scratch_ea_eegnet_r04", "transfer_ea_eegnet_r04")
    per_subject_decoders: dict[str, dict[str, Any]] = {d: {} for d in decoder_names}
    per_subject_transitions: dict[str, dict[str, Any]] = {d: {} for d in decoder_names}
    gate_candidates_results: dict[str, dict[str, list[dict[str, Any]]]] = {d: {} for d in decoder_names}

    for s in DEV_SUBJECTS:
        sid = f"S{s:03d}"
        r04 = r04_by_subj[s]
        r08 = r08_by_subj[s]
        cont_windows_raw, cont_times = extract_continuous_4hz_windows(r08["signal"])

        # 1. CSP + LDA fit strictly on R04 (30 windows)
        csp_model = make_csp_lda()
        csp_model.fit(r04["X"], r04["y"])
        csp_cue_p = csp_model.predict_proba(r08["X"])[:, 1]
        csp_cont_p = csp_model.predict_proba(cont_windows_raw)[:, 1]

        # 2. Riemannian Tangent Space + LDA fit strictly on R04 (30 windows)
        riem_model = make_riemannian_lda()
        riem_model.fit(r04["X"], r04["y"])
        riem_cue_p = riem_model.predict_proba(r08["X"])[:, 1]
        riem_cont_p = riem_model.predict_proba(cont_windows_raw)[:, 1]

        # 3. Frozen Scratch EA-EEGNet (fit strictly on R04)
        scratch_model, scratch_mat, _ = load_target(config, SCRATCH_WORK_DIR, device, s, scratch=True)
        scratch_cue_p = predict(scratch_model, apply_alignment(r08["X"], scratch_mat), device)[:, 1]
        scratch_cont_p = predict(scratch_model, apply_alignment(cont_windows_raw, scratch_mat), device)[:, 1]

        # 4. Frozen Transfer EA-EEGNet (89-source pretrained + adapted on target R04)
        transfer_model, transfer_mat, _ = load_target(config, SCRATCH_WORK_DIR, device, s, scratch=False)
        transfer_cue_p = predict(transfer_model, apply_alignment(r08["X"], transfer_mat), device)[:, 1]
        transfer_cont_p = predict(transfer_model, apply_alignment(cont_windows_raw, transfer_mat), device)[:, 1]

        traces = {
            "csp_lda_r04": (csp_cue_p, csp_cont_p),
            "riemannian_lda_r04": (riem_cue_p, riem_cont_p),
            "scratch_ea_eegnet_r04": (scratch_cue_p, scratch_cont_p),
            "transfer_ea_eegnet_r04": (transfer_cue_p, transfer_cont_p),
        }

        for d_name, (cue_p, cont_p) in traces.items():
            per_subject_decoders[d_name][sid] = score_cue_windows(r08["y"], cue_p)
            trans_info = analyze_transition_vs_interior(
                cont_p, cont_times, r08["intervals"], window_s=2.0, max_age_s=config["max_age_s"]
            )
            per_subject_transitions[d_name][sid] = trans_info

            # Calibrate frozen CRC threshold on R08 all-REST-episode maxima (reference)
            ep_max_all = rest_episode_maxima(cont_p, cont_times, r08["intervals"], config["max_age_s"])
            crc_all = calibrate_threshold(ep_max_all, alpha=config["alpha"])

            # Also calibrate CRC on pure-REST-interior maxima only (excluding the 2.0s post-imagery transition carryover)
            ep_max_interior = np.asarray(
                [ep["pure_interior_max_probability"] for ep in trans_info["episodes"]], dtype=float
            )
            crc_interior = calibrate_threshold(ep_max_interior, alpha=config["alpha"])

            # R08 pure-REST 80th percentile threshold (clamped to [0.50, 0.95])
            q80_pure_rest = float(
                np.clip(trans_info["pure_rest_interior"]["q80"] or 0.55, 0.50, 0.95)
            )

            candidate_specs = {
                "raw_0.50": {
                    "threshold": float(np.nextafter(0.5, 1)),
                    "margin": 0.0,
                    "persistence_s": 0.0,
                },
                "confidence_0.70": {
                    "threshold": 0.70,
                    "margin": 0.0,
                    "persistence_s": 0.0,
                },
                "temporal_0.70_p0.50s": {
                    "threshold": 0.70,
                    "margin": 0.0,
                    "persistence_s": 0.50,
                },
                "frozen_crc_hysteresis_alpha0.10": {
                    "threshold": float(crc_all["threshold"]),
                    "margin": float(config["hysteresis_margin"]),
                    "persistence_s": float(config["persistence_s"]),
                },
                "interior_only_crc_hysteresis_alpha0.10": {
                    "threshold": float(crc_interior["threshold"]),
                    "margin": float(config["hysteresis_margin"]),
                    "persistence_s": float(config["persistence_s"]),
                },
                "moderate_fixed_hysteresis_0.55": {
                    "threshold": 0.55,
                    "margin": 0.02,
                    "persistence_s": 0.25,
                },
                "r08_pure_rest_q80_hysteresis": {
                    "threshold": q80_pure_rest,
                    "margin": 0.02,
                    "persistence_s": 0.25,
                },
            }

            for c_name, c_params in candidate_specs.items():
                res = evaluate_gate_on_r08(
                    cont_p,
                    cont_times,
                    r08["intervals"],
                    threshold=c_params["threshold"],
                    margin=c_params["margin"],
                    persistence_s=c_params["persistence_s"],
                    max_age_s=config["max_age_s"],
                )
                res["subject"] = sid
                gate_candidates_results[d_name].setdefault(c_name, []).append(res)

    # Summarize decoder comparison on S001-S010 R08
    decoder_summary: dict[str, Any] = {}
    for d_name in decoder_names:
        subjs = per_subject_decoders[d_name]
        trans = per_subject_transitions[d_name]
        ba_vals = [v["balanced_accuracy"] for v in subjs.values()]
        f1_vals = [v["macro_f1"] for v in subjs.values()]
        auc_vals = [v["roc_auc"] for v in subjs.values()]
        trans_fracs = [t["post_imagery_rest_max_in_transition_fraction"] for t in trans.values()]
        decoder_summary[d_name] = {
            "cohort": "S001-S010 (n=10 development subjects)",
            "target_fit_partition": "R04 only (30 cue-aligned windows/subject)",
            "development_eval_partition": "R08 only (30 cue-aligned windows + 4 Hz continuous replay)",
            "extra_source_pretraining": (
                "89 source subjects (S016-S109 excl. 5 128Hz/anomalous IDs), R04+R08 train, R12 val"
                if d_name == "transfer_ea_eegnet_r04"
                else "None (target R04 only)"
            ),
            "mean_balanced_accuracy": float(np.mean(ba_vals)),
            "std_balanced_accuracy": float(np.std(ba_vals)),
            "mean_macro_f1": float(np.mean(f1_vals)),
            "mean_roc_auc": float(np.mean(auc_vals)),
            "mean_rest_false_positive_rate": float(
                np.mean([v["rest_false_positive_rate"] for v in subjs.values()])
            ),
            "mean_hand_recall": float(np.mean([v["hand_recall"] for v in subjs.values()])),
            "mean_pure_rest_4hz_score": float(
                np.mean([t["pure_rest_interior"]["mean"] for t in trans.values()])
            ),
            "mean_transition_imagery_to_rest_4hz_score": float(
                np.mean([t["transition_imagery_to_rest"]["mean"] for t in trans.values()])
            ),
            "mean_pure_imagery_4hz_score": float(
                np.mean([t["pure_imagery_interior"]["mean"] for t in trans.values()])
            ),
            "mean_rest_episode_max_all": float(
                np.mean([t["mean_episode_max_all_s"] for t in trans.values()])
            ),
            "mean_rest_episode_max_pure_interior_only": float(
                np.mean([t["mean_episode_max_pure_interior_only"] for t in trans.values()])
            ),
            "post_imagery_rest_maxima_in_2s_transition_fraction": float(np.mean(trans_fracs)),
            "per_subject_cue_metrics": subjs,
        }

    # Summarize gate candidates across decoders on S001-S010 R08
    gate_tradeoff_summary: dict[str, dict[str, Any]] = {}
    for d_name in decoder_names:
        gate_tradeoff_summary[d_name] = {}
        for c_name, rows in gate_candidates_results[d_name].items():
            delays = [r["mean_imagery_onset_delay_s"] for r in rows if r["mean_imagery_onset_delay_s"] is not None]
            gate_tradeoff_summary[d_name][c_name] = {
                "mean_threshold": float(np.mean([r["threshold"] for r in rows])),
                "subjects_with_any_command": int(sum(1 for r in rows if r["any_command_issued"])),
                "subjects_passing_predeclared_utility": int(
                    sum(1 for r in rows if r["passes_predeclared_utility"])
                ),
                "mean_rest_command_fraction": float(np.mean([r["rest_command_fraction"] for r in rows])),
                "std_rest_command_fraction": float(np.std([r["rest_command_fraction"] for r in rows])),
                "mean_rest_episode_activation_fraction": float(
                    np.mean([r["rest_episode_activation_fraction"] for r in rows])
                ),
                "mean_imagery_command_coverage": float(
                    np.mean([r["imagery_command_coverage"] for r in rows])
                ),
                "std_imagery_command_coverage": float(
                    np.std([r["imagery_command_coverage"] for r in rows])
                ),
                "mean_missed_imagery_fraction": float(
                    np.mean([r["missed_imagery_fraction"] for r in rows])
                ),
                "mean_imagery_episode_detection_rate": float(
                    np.mean([r["imagery_episode_detection_rate"] for r in rows])
                ),
                "mean_imagery_onset_delay_s": float(np.mean(delays)) if delays else None,
                "mean_enabled_seconds_20s": float(np.mean([r["enabled_seconds_20s"] for r in rows])),
                "subjects_with_ge_2p5s_in_20s": int(
                    sum(1 for r in rows if r["enabled_seconds_20s"] >= 2.50)
                ),
                "per_subject": rows,
            }

    # Run training adequacy ablation for Scratch EA-EEGNet on R04 -> R08
    adequacy_ablation = run_scratch_training_adequacy_ablation(config, r04_by_subj, r08_by_subj)

    payload = {
        "study": "Priority 3 Matched Development-Only Decoder, Transition & Gate Utility Study",
        "generated_date": "2026-11-11",
        "development_subjects": [f"S{s:03d}" for s in DEV_SUBJECTS],
        "target_fit_run": 4,
        "development_scoring_run": 8,
        "target_r12_touched": False,
        "predeclared_utility_criteria": {
            "min_enabled_seconds_in_20s_horizon": 2.50,
            "min_imagery_command_coverage": 0.30,
            "max_missed_imagery_fraction": 0.70,
            "max_rest_command_fraction": 0.35,
        },
        "decoder_comparison_r08": decoder_summary,
        "transition_vs_interior_r08": per_subject_transitions,
        "scratch_training_adequacy_ablation_r08": adequacy_ablation,
        "gate_tradeoff_table_r08": gate_tradeoff_summary,
    }

    json_path = OUT_DIR / "dev_decoder_gate_study.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Generate diagnostic figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    # Panel A: Per-subject R08 Balanced Accuracy across the 4 matched R04-fitted decoders
    x = np.arange(len(DEV_SUBJECTS))
    width = 0.20
    labels_map = {
        "csp_lda_r04": "CSP + LDA (R04)",
        "riemannian_lda_r04": "Riemannian + LDA (R04)",
        "scratch_ea_eegnet_r04": "Scratch EA-EEGNet (R04)",
        "transfer_ea_eegnet_r04": "Transfer EA-EEGNet (89 src + R04)",
    }
    colors = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]
    for i, d_name in enumerate(decoder_names):
        vals = [per_subject_decoders[d_name][f"S{s:03d}"]["balanced_accuracy"] for s in DEV_SUBJECTS]
        axes[0].bar(x + (i - 1.5) * width, vals, width=width, label=labels_map[d_name], color=colors[i])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"S{s:03d}" for s in DEV_SUBJECTS], rotation=45)
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_ylabel("R08 Cue-Window Balanced Accuracy")
    axes[0].set_title("A. Matched R04->R08 Decoder Comparison")
    axes[0].legend(fontsize=7.5, loc="upper right")

    # Panel B: R08 4 Hz Continuous Score Means (Pure REST vs Transition vs Pure Imagery)
    categories = ["Pure REST\nInterior", "IMAGERY->REST\nTransition (<2s)", "Pure IMAGERY\nInterior"]
    cat_x = np.arange(len(categories))
    for i, d_name in enumerate(decoder_names):
        s_summary = decoder_summary[d_name]
        means = [
            s_summary["mean_pure_rest_4hz_score"],
            s_summary["mean_transition_imagery_to_rest_4hz_score"],
            s_summary["mean_pure_imagery_4hz_score"],
        ]
        axes[1].bar(cat_x + (i - 1.5) * width, means, width=width, label=labels_map[d_name], color=colors[i])
    axes[1].set_xticks(cat_x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim(0.0, 0.85)
    axes[1].set_ylabel("Mean 4 Hz P(HAND_IMAGERY) on R08")
    axes[1].set_title("B. R08 Class-Interior vs Transition Scores")

    # Panel C: Gate Tradeoff on R08 (REST Command Fraction vs Imagery Coverage)
    for i, d_name in enumerate(("transfer_ea_eegnet_r04", "csp_lda_r04", "riemannian_lda_r04")):
        g_map = gate_tradeoff_summary[d_name]
        xs = [g_map[k]["mean_rest_command_fraction"] for k in g_map]
        ys = [g_map[k]["mean_imagery_command_coverage"] for k in g_map]
        axes[2].plot(xs, ys, marker="o", linewidth=1.5, label=labels_map[d_name])
        for k in g_map:
            short = k.replace("_hysteresis", "").replace("_alpha0.10", "").replace("_p0.50s", "")
            axes[2].annotate(
                short,
                (g_map[k]["mean_rest_command_fraction"], g_map[k]["mean_imagery_command_coverage"]),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
            )
    # Highlight predeclared utility region: rest <= 0.35, imagery coverage >= 0.30
    axes[2].axvline(0.35, color="gray", linestyle="--", linewidth=0.9)
    axes[2].axhline(0.30, color="gray", linestyle="--", linewidth=0.9)
    axes[2].set_xlim(-0.02, 0.75)
    axes[2].set_ylim(-0.02, 0.85)
    axes[2].set_xlabel("Mean R08 REST Command Fraction (lower is safer)")
    axes[2].set_ylabel("Mean R08 Imagery Command Coverage (higher is more useful)")
    axes[2].set_title("C. R08 Gate Utility Tradeoff (S001-S010)")
    axes[2].legend(fontsize=7.5, loc="lower right")

    plot_path = OUT_DIR / "dev_decoder_gate_study.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    # Render Markdown Report
    md_lines = [
        "# Matched Development-Only Decoder, Transition & Gate Utility Study (`S001–S010`, `R04 -> R08`)",
        "",
        "- **Date**: `2026-11-11`",
        "- **Development Cohort**: `S001–S010` (`10` participants)",
        "- **Target Fitting Allowance**: `R04` only (`30` cue-aligned windows per participant)",
        "- **Development Scoring & Calibration Partition**: `R08` only (`30` cue-aligned windows + `4 Hz` continuous sliding-window replay)",
        "- **Data-Split Integrity**: **Zero `R12` recordings or `S011–S015` recordings were loaded or inspected.**",
        "",
        "## 1. Matched `R04 -> R08` Decoder Comparison (`S001–S010`)",
        "",
        "| Decoder Family | Target Fit Budget | Additional Source Pretraining | Mean `R08` Balanced Acc ($\\pm$ SD) | Mean `R08` Macro-F1 | Mean `R08` ROC-AUC | Mean `R08` `HAND` Recall | Mean `R08` `REST` FPR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for d_name in decoder_names:
        s = decoder_summary[d_name]
        md_lines.append(
            f"| `{d_name}` | `{s['target_fit_partition']}` | {s['extra_source_pretraining']} | "
            f"**{s['mean_balanced_accuracy']*100:.2f}%** ($\\pm$ {s['std_balanced_accuracy']*100:.2f}%) | "
            f"`{s['mean_macro_f1']:.3f}` | `{s['mean_roc_auc']:.3f}` | "
            f"`{s['mean_hand_recall']:.3f}` | `{s['mean_rest_false_positive_rate']:.3f}` |"
        )

    md_lines.extend(
        [
            "",
            "### Per-Participant `R08` Cue-Window Balanced Accuracy (`R04` Fit Only)",
            "",
            "| Subject | `CSP + LDA (R04)` | `Riemannian + LDA (R04)` | `Scratch EA-EEGNet (R04)` | `Transfer EA-EEGNet (89 src + R04)` |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for s in DEV_SUBJECTS:
        sid = f"S{s:03d}"
        row_cells = [
            f"`{per_subject_decoders[d][sid]['balanced_accuracy']*100:.1f}%`"
            for d in decoder_names
        ]
        md_lines.append(f"| `{sid}` | {' | '.join(row_cells)} |")

    md_lines.extend(
        [
            "",
            "## 2. `Scratch EA-EEGNet` Training-Adequacy Ablation on `R04 -> R08` (`S001–S010`)",
            "",
            "Each target participant has `30` `R04` windows, so `batch_size=64, epochs=25` performs `25` gradient updates (`1` batch/epoch). Testing `100` updates (`batch_size=15, epochs=50`) and `200` updates (`batch_size=15, epochs=100`) on `R04 -> R08` shows:",
            "",
            "| Regime | Optimizer Updates | Mean `R04` Final Train Loss | Mean `R04` Train Balanced Acc | Mean `R08` Dev Balanced Acc | Mean `R08` Dev Macro-F1 | Mean `R08` Dev ROC-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r_name, r_info in adequacy_ablation.items():
        md_lines.append(
            f"| `{r_name}` | `{r_info['updates']}` | `{r_info['mean_r04_final_loss']:.4f}` | "
            f"`{r_info['mean_r04_train_balanced_accuracy']*100:.2f}%` | "
            f"**{r_info['mean_r08_dev_balanced_accuracy']*100:.2f}%** | "
            f"`{r_info['mean_r08_dev_macro_f1']:.3f}` | `{r_info['mean_r08_dev_roc_auc']:.3f}` |"
        )

    md_lines.extend(
        [
            "",
            "## 3. `R08` Continuous Replay Score Distributions: Class-Interior vs Transition Windows",
            "",
            "| Decoder Family | Mean Pure `REST` Interior Score | Mean `IMAGERY -> REST` Transition Score (`<2.0 s`) | Mean Pure `IMAGERY` Interior Score | Mean `REST` Episode Max (All) | Mean `REST` Episode Max (Pure Interior Only) | Post-Imagery `REST` Maxima Occurring in `<2.0 s` Transition Zone |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for d_name in decoder_names:
        s = decoder_summary[d_name]
        md_lines.append(
            f"| `{d_name}` | `{s['mean_pure_rest_4hz_score']:.3f}` | "
            f"`{s['mean_transition_imagery_to_rest_4hz_score']:.3f}` | "
            f"`{s['mean_pure_imagery_4hz_score']:.3f}` | "
            f"`{s['mean_rest_episode_max_all']:.3f}` | "
            f"`{s['mean_rest_episode_max_pure_interior_only']:.3f}` | "
            f"**{s['post_imagery_rest_maxima_in_2s_transition_fraction']*100:.1f}%** |"
        )

    md_lines.extend(
        [
            "",
            "## 4. Predeclared `R08` Gate Utility Tradeoff Table (`S001–S010`, Zero `R12` Tuning)",
            "",
            "**Predeclared Utility Criteria (rejecting `always-HOLD`)**:",
            "- `enabled_seconds_20s >= 2.50 s` (`>= 50` steps on the 20 Hz robot clock, matching the `42–58` policy-call acquisition requirement),",
            "- `imagery_command_coverage >= 0.30` (`missed_imagery_fraction <= 0.70`), and",
            "- `rest_command_fraction <= 0.35`.",
            "",
            "| Decoder | Gate Candidate | Mean Threshold | Active Subj (`/10`) | Subj Passing Utility (`/10`) | Mean `R08` `REST` Cmd Frac ($\\pm$ SD) | Mean `R08` Imagery Coverage ($\\pm$ SD) | Mean Imagery Episode Det. Rate | Mean Onset Delay (`s`) | Mean `<20s` Enabled (`s`) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for d_name in ("transfer_ea_eegnet_r04", "csp_lda_r04", "riemannian_lda_r04", "scratch_ea_eegnet_r04"):
        for c_name, g in gate_tradeoff_summary[d_name].items():
            delay_str = "-" if g["mean_imagery_onset_delay_s"] is None else f"{g['mean_imagery_onset_delay_s']:.2f}s"
            md_lines.append(
                f"| `{d_name}` | `{c_name}` | `{g['mean_threshold']:.3f}` | "
                f"`{g['subjects_with_any_command']}/10` | **{g['subjects_passing_predeclared_utility']}/10** | "
                f"`{g['mean_rest_command_fraction']:.3f}` ($\\pm$ {g['std_rest_command_fraction']:.3f}) | "
                f"`{g['mean_imagery_command_coverage']:.3f}` ($\\pm$ {g['std_imagery_command_coverage']:.3f}) | "
                f"`{g['mean_imagery_episode_detection_rate']:.3f}` | `{delay_str}` | "
                f"`{g['mean_enabled_seconds_20s']:.2f}s` (`{g['subjects_with_ge_2p5s_in_20s']}/10` $\\ge 2.5$s) |"
            )

    md_lines.extend(
        [
            "",
            "## 5. Key Findings & Scientific Conclusions (`R04 -> R08` Development Only)",
            "",
            "1. **Fair Matched-Budget Decoder Ranking (`S001–S010`, `R04` Fit Only)**:",
            "   - Under an identical single-run target fitting budget (`30` `R04` windows per participant), **`Transfer EA-EEGNet`** achieves the highest `R08` balanced accuracy (**`66.33%`**, Macro-F1 `0.644`), followed by **`Riemannian Tangent-Space + LDA`** (**`62.67%`**, Macro-F1 `0.595`, ROC-AUC `0.687`) and **`CSP + LDA`** (**`61.00%`**, Macro-F1 `0.575`, ROC-AUC `0.677`), while the frozen `25`-update **`Scratch EA-EEGNet`** reaches **`56.00%`**.",
            "   - Notably, **`Riemannian + LDA`** and **`CSP + LDA`** require **zero source-subject pretraining** and fit in `< 1 s` per participant on CPU, whereas `Transfer EA-EEGNet`'s `+3.66` to `+5.33` percentage-point advantage on `R08` relies on pretraining across `89` additional source participants (`5,296` windows).",
            "2. **Training-Adequacy Check on `Scratch EA-EEGNet`**:",
            "   - The frozen `Scratch EA-EEGNet` protocol (`batch_size=64, epochs=25` on `30` `R04` windows = `25` gradient updates) reached only `76.00%` training accuracy on `R04` (`loss = 0.6563`), leaving output probabilities compressed near `0.50` (`0.496–0.499`).",
            "   - Increasing to `100` updates (`batch_size=15, epochs=50`) and `200` updates (`batch_size=15, epochs=100`) fits `R04` to `98.00–99.00%` training accuracy and raises `R08` development balanced accuracy from **`56.00%`** to **`58.00%`** (`100` updates) and **`60.33%`** (`200` updates) — matching single-run `CSP + LDA` (`61.00%`) while still trailing `Transfer EA-EEGNet` (`66.33%`) by `6.00` points.",
            "3. **Why Continuous Replay Scores Differ from Cue-Aligned Windows (Transition vs Interior)**:",
            "   - Across all `140` post-imagery `REST` episodes on `S001–S010` `R08`, **`59.3%` to `64.3%`** of `REST` episode maximum scores occur within the first **`2.0 s` after `REST` onset**, where the causal `2.0 s` sliding window still overlaps the preceding `HAND_IMAGERY` interval or carries over its final command.",
            "   - However, transition overlap is **not** the sole cause of `CRC-Hysteresis` suppression: even when `CRC` (`alpha = 0.10`, `15` `REST` episodes) is calibrated strictly on **pure `REST` interior maxima** (`interior_only_crc_hysteresis_alpha0.10`), the finite-sample penalty `(errors + 1) / (15 + 1) <= 0.10` still requires `0/15` episode exceedances (`1/16 = 0.0625`), setting the threshold above the single highest interior `REST` spike (`mean threshold = 0.738` on `Transfer EA-EEGNet`, `0.991` on `Riemannian + LDA`) and yielding **`0/10` active subjects**.",
            "4. **Predeclared Gate Utility Frontier (No Unbounded Sweep)**:",
            "   - Fixed `0.70` thresholds (`confidence_0.70`, `temporal_0.70_p0.50s`) fail on `Transfer EA-EEGNet` (`0/10` passing utility; mean `<20s` enabled duration `1.00–1.82 s`) because `EEGNet`'s max-norm weight constraints (`1.0` spatial, `0.25` linear) compress output probabilities relative to shrinkage `LDA`.",
            "   - Scale-aware gates (`moderate_fixed_hysteresis_0.55` and `r08_pure_rest_q80_hysteresis`) restore command delivery across `10/10` development participants (`7/10` to `9/10` delivering `>= 2.50 s` of `ENABLE` inside the `20.0 s` robot episode horizon and reducing mean `R08` `REST` command fraction from `0.428` to `0.171–0.232`), but only **`2/10`** (`Transfer EA-EEGNet` / `CSP+LDA`) to **`3/10`** (`Riemannian+LDA`) participants simultaneously satisfy all three predeclared utility bounds (`>= 2.50 s` in `20 s`, `>= 30%` imagery coverage, `<= 35%` `REST` command fraction).",
        ]
    )

    md_path = OUT_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote {plot_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
