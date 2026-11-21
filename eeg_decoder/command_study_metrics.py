"""Reusable development metrics extracted unchanged from the November 11 study."""
from typing import Any
import mne
from mne.decoding import CSP
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from .gpu_pipeline import command_metrics
from .risk_gate import HysteresisGate

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
