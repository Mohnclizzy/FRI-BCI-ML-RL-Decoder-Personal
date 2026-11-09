"""Causal hysteresis and CRC-adjusted rest-episode score calibration.

The threshold is fitted to episode maxima, not overlapping windows. The
finite-sample correction is conditional on exchangeable calibration/test
episodes and a fixed predictor. Single-recording EEG does not establish those
assumptions: reports explicitly do NOT assert prospective risk guarantees.
"""
import math
import numpy as np


class HysteresisGate:
    def __init__(self, threshold, margin=0.05, persistence_s=0.5, max_age_s=1.0):
        if not all(math.isfinite(v) for v in (threshold, margin, persistence_s, max_age_s)):
            raise ValueError('Parameters must be finite.')
        if threshold < 0.5 or threshold > 1.1 or margin < 0 or persistence_s < 0 or max_age_s <= 0:
            raise ValueError('Invalid gate parameters.')
        self.threshold, self.margin = threshold, margin
        self.persistence_s, self.max_age_s = persistence_s, max_age_s
        self.reset()

    def reset(self):
        self.state, self.since, self.last_end, self.last_available = 'hold', None, None, None
        self.key = None

    def update(self, probability, window_end_s, available_at_s, now_s, key):
        values = (probability, window_end_s, available_at_s, now_s)
        if not all(math.isfinite(v) for v in values) or not 0 <= probability <= 1:
            raise ValueError('Invalid probability or timestamp.')
        if window_end_s < 0 or available_at_s < window_end_s:
            raise ValueError('Noncausal estimate.')
        if self.key != key:
            self.reset(); self.key = key
        def hold(reason):
            self.state, self.since = 'hold', None
            return {'skill': 'hold', 'reason': reason, 'issued_at_s': now_s, 'state': self.state}
        if now_s < available_at_s:
            return hold('not_yet_available')
        if now_s - window_end_s > self.max_age_s:
            return hold('stale')
        if self.last_end is not None and (window_end_s <= self.last_end or available_at_s < self.last_available):
            return hold('duplicate_or_out_of_order')
        if self.last_end is not None and now_s - self.last_end > self.max_age_s:
            self.state, self.since = 'hold', None
        self.last_end, self.last_available = window_end_s, available_at_s
        if self.threshold > 1 or probability < self.threshold:
            return hold('disabled' if self.threshold > 1 else 'below_hold_threshold')
        if self.state != 'enable':
            # Raising the calibrated threshold raises BOTH activation and hold
            # thresholds. Active commands are always a subset of p >= threshold.
            if probability < min(1.0, self.threshold + self.margin):
                return hold('below_enable_threshold')
            if self.since is None:
                self.since = now_s
            self.state = 'collecting'
            if now_s - self.since + 1e-9 < self.persistence_s:
                return {'skill': 'hold', 'reason': 'collecting_evidence', 'issued_at_s': now_s, 'state': self.state}
        self.state = 'enable'
        return {'skill': 'grasp_lift', 'reason': 'accepted', 'issued_at_s': now_s, 'state': self.state}

    def tick(self, now_s):
        if not math.isfinite(now_s):
            raise ValueError('Invalid clock.')
        if self.last_end is None or now_s - self.last_end > self.max_age_s or now_s < self.last_available:
            self.state, self.since = 'hold', None
        return 'grasp_lift' if self.state == 'enable' else 'hold'


def rest_episode_maxima(probabilities, times, intervals, max_age_s):
    """Include carried scores from before REST onset while commands are fresh.

    Inference is instantaneous offline (available_at == window_end). Each
    prediction controls [time, min(next_time, time+max_age)], clipped to the
    annotation. Labels are used ONLY for this calibration/evaluation function.
    """
    p, t = np.asarray(probabilities, float), np.asarray(times, float)
    if p.ndim != 1 or p.shape != t.shape or not len(t) or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Invalid score trace.')
    if (np.diff(t) <= 0).any() or ((p < 0) | (p > 1)).any() or max_age_s <= 0:
        raise ValueError('Unordered scores or invalid parameters.')
    ends = np.minimum(np.r_[t[1:], np.inf], t + max_age_s)
    scores = []
    for start, end, label in intervals:
        if label != 0:
            continue
        overlap = (t < end) & (ends > start)
        scores.append(float(p[overlap].max()) if overlap.any() else 0.0)
    return np.asarray(scores)


def calibrate_threshold(episode_scores, alpha=0.1):
    """Smallest fixed-grid threshold with (sum(loss)+1)/(n+1) <= alpha.

    Loss is any above-threshold score during a REST episode, an upper envelope
    for gate activation in that episode. The no-enable fallback is structural,
    not a claimed certificate inferred from an undersized calibration set.
    """
    scores = np.asarray(episode_scores, float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Need finite REST episode maxima in [0, 1].')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1).')
    curve = []
    chosen = None
    for threshold in np.linspace(0.5, 1.0, 101):
        errors = int((scores >= threshold).sum())
        corrected = (errors + 1) / (len(scores) + 1)
        curve.append({'threshold': float(threshold), 'episode_exceedances': errors, 'crc_adjusted_risk': corrected})
        if chosen is None and corrected <= alpha:
            chosen = curve[-1]
    return {
        'threshold': chosen['threshold'] if chosen else 1.1,
        'status': 'empirical_crc_candidate' if chosen else 'always_hold_fallback',
        'alpha': alpha, 'n_rest_episodes': len(scores),
        'minimum_finite_sample_correction': 1 / (len(scores) + 1),
        'selected_crc_adjusted_risk': chosen['crc_adjusted_risk'] if chosen else None,
        'risk_unit': 'any above-threshold fresh score within an annotated REST episode',
        'guarantee_claimed': False,
        'limitations': 'Within-run episodes can be dependent; run shift and exchangeability are unverified. No physical safety or motion-risk guarantee.',
        'curve': curve,
    }
