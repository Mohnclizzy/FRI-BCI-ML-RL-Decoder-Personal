# Antigravity CLI review (unverified advisory output)

Antigravity CLI 1.5.6, noninteractive review using supplied code and R08 diagnostics. The first filesystem-based attempt was denied by CLI command permissions; no bypass was used. This successful review was instructed not to use tools. Read gate_diagnosis.md and the adjudication below before treating suggestions as conclusions.

### Executive Summary

The frozen Conformal Risk Control (CRC) + hysteresis gate suppressed 100% of commands across all 30 evaluated models (15 scratch, 15 transfer) because **finite-sample CRC calibration combined with a tiny calibration set ($n=15$) mathematically forced a zero-tolerance threshold ($\text{errors} = 0$), which was subsequently compounded by an additive margin and multi-frame temporal persistence**. 

The code contains no runtime crashes or logical coding bugs; rather, the complete failure to emit commands stems from **severe structural over-conservatism in the calibration design** coupled with **poor decoder confidence (especially in scratch models)** and **uncalibrated output probabilities**.

---

### 1. Why the Gate Suppresses Every Command

#### A. Mathematical Chokepoint in CRC Calibration
In `calibrate_threshold`, the empirical finite-sample adjusted risk is computed as:
$$\widehat{R}(t) = \frac{\sum_{i=1}^n \text{loss}_i + 1}{n + 1} = \frac{\text{errors} + 1}{16}$$
Given $\alpha = 0.10$ and $n = 15$ REST episodes:
- If $\text{errors} = 0$: $\widehat{R} = \frac{1}{16} = 0.0625 \le 0.10$ (Accepted)
- If $\text{errors} = 1$: $\widehat{R} = \frac{2}{16} = 0.1250 > 0.10$ (Rejected)

Therefore, **the algorithm cannot tolerate a single REST exceedance during R08 calibration**. The base threshold is forced to exceed the maximum score observed across all 15 REST episodes:
$$\text{threshold} > \max_{i \in \text{REST}} (\text{score}_i)$$

#### B. Triple-Compounded Gate Hurdles
To emit an active command (`grasp_lift`), a prediction must clear three sequential hurdles:
1. **Conservative Base Threshold ($p \ge \text{threshold}$):** Set above the maximum REST score.
2. **Additive Hysteresis Margin ($p \ge \text{threshold} + 0.05$):** Required to transition out of `hold` / initiate `collecting`.
3. **Temporal Persistence ($\ge 0.5\text{ s}$ at $0.25\text{ s}$ stride):** At least 3 consecutive samples spanning $\ge 0.5\text{ s}$ ($t_0, t_0+0.25, t_0+0.5$) must continuously satisfy $p \ge \text{threshold} + 0.05$.

#### C. Evidence from R08 Audit Data
- **Scratch Models (15/15):** Severely underfitted on Target R04. Max probabilities across all evaluation frames ranged from **0.502 to 0.551** (barely above chance). Because enable thresholds were set to $0.555 - 0.590$, **zero samples ever reached the enable threshold**.
- **Transfer Models (15/15):** While decoders output higher probabilities, resting EEG artifacts/noise pushed R08 calibration thresholds up to $0.685 - 0.870$ (enable thresholds $0.735 - 0.920$).
  - For 14 out of 15 transfer models, `max_score` never reached `enable_threshold`.
  - For **Transfer S001**, `max_score` reached $0.8672$ against an enable threshold of $0.860$, but only for **1 single sample** (`longest_enable_streak_samples = 1`). Because the gate requires $0.5\text{ s}$ persistence ($2\text{--}3$ consecutive samples), S001 remained trapped in `collecting_evidence` (`hold`) before decaying.

---

### 2. Distinguishing Bugs from Conservative Calibration

| Element | Classification | Mechanism & Justification |
| :--- | :--- | :--- |
| **State Machine Execution** | **Not a Bug** | The causal checks, staleness logic (`max_age_s`), timestamp validation, and state transitions (`hold` $\rightarrow$ `collecting` $\rightarrow$ `enable`) execute faithfully according to the specification. |
| **CRC Formula Implementation** | **Not a Bug** | $\frac{\text{errors}+1}{n+1} \le \alpha$ accurately mirrors the exchangeable finite-sample conformal risk bound for bounded step loss. |
| **Sample Size / $\alpha$ Mismatch** | **Design Flaw (Extreme Conservatism)** | Setting $n=15$ and $\alpha=0.1$ creates a structural floor of $0.0625$. The threshold is forced to the sample maximum ($\text{errors} = 0$), transforming the calibration into an extreme value statistic susceptible to single-outlier corruption. |
| **Redundant Conservatism (Double-Penalty)** | **Design Flaw (Mismatched Loss)** | `calibrate_threshold` uses raw point score exceedance in a REST episode as its loss function, which assumes an instantaneous gate. The runtime gate then independently layers a $+0.05$ margin and $0.5\text{ s}$ temporal filter on top. The gate penalizes the decoder twice for false-activation risk. |
| **Scratch Model Output Collapse** | **Modeling / Underfitting Defect** | Target R04 fit is insufficient for the scratch architecture to learn discriminative features, yielding chance predictions ($\sim 0.50 - 0.55$). |
| **Non-Exchangeability / Run Shift** | **Methodological Limitation** | EEG background spectra and impedance drift between calibration (R08) and evaluation runs. A single transient burst during R08 inflates the threshold, locking out valid activations downstream. |

---

### 3. Development-Only Diagnostic Protocol (No Target R12 Tuning)

> [!IMPORTANT]
> **Strict Partitioning Boundary:** Target R12 was already evaluated and is exploratory; it must **never** be accessed for diagnostic tuning, hyperparameter search, or threshold selection. All diagnostics and recalibrations must be performed exclusively on **Source S016–S109 (R04/R08/R12)** and **Target S001–S015 (R04/R08)**.

```
DEVELOPMENT PARTITIONS ONLY
┌─────────────────────────────────────────────────────────────────┐
│ Source Cohort (S016–S109, excl. 5):                             │
│   ├── R04 / R08: Feature extraction & Decoder Pre-training      │
│   └── R12: Architecture Validation & Gate Dynamic Tuning        │
├─────────────────────────────────────────────────────────────────┤
│ Target Cohort (S001–S015):                                      │
│   ├── R04: Internal K-Fold CV (Scratch Fitting & Adaptation)    │
│   └── R08: Split Calibration & Threshold Audit                  │
│   [Target R12: FROZEN / UNTOUCHED EXPLORATORY EVALUATION]       │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 1: Probability Calibration & Decoder Representation Audit (Target R04 & Source Dev)
- **Scratch Underfitting Analysis:** Using nested cross-validation exclusively within Target R04, determine why scratch models output near-uniform probabilities ($\le 0.55$). Evaluate whether class imbalance, regularized CSP/Riemannian covariance shrinkage, or filter-bank feature extraction restores separation between REST and ACTIVE windows.
- **Probability Calibration (Platt / Isotonic / Temperature Scaling):** Fit temperature scaling on Source R12 validation outputs to ensure predicted probabilities match empirical posteriors, preventing uncalibrated high scores during rest.

#### Step 2: Align the CRC Loss Function with the Gate State Machine (Source R12 Dev)
- **Simulated Gate Loss:** In `calibrate_threshold`, replace the raw window-level loss (`scores >= threshold`) with an end-to-end **gate-activation loss** computed by passing the calibration run through the full `HysteresisGate(threshold, margin, persistence_s)`.
  - An episode error should occur **only if the gate transitions to `enable` (`grasp_lift`)** during that REST episode.
  - This removes the double penalty: the temporal persistence and margin are accounted for *during* threshold selection, rather than stacked arbitrarily on top of an already zero-exceedance threshold.

#### Step 3: Granularity & Sample Partitioning for Calibration (Target R08 Dev)
- **Sub-episode Segmentation:** With $n=15$, $(e+1)/16$ forces $e=0$. On development partitions, examine whether dividing long REST periods into shorter standardized sub-episodes (e.g., $2.0\text{ s}$ epochs, preserving temporal autocorrelation blocks) increases $n$ (e.g., $n=45 \implies (e+1)/46 \le 0.1 \implies e \le 3$). This prevents a single transient artifact from forcing the threshold to the global maximum.
- **Run-to-Run Non-Stationarity Audit:** Quantify distribution shift between R04 and R08 REST distributions (Wasserstein distance or Kolmogorov-Smirnov test on development data) to detect baseline drift.

#### Step 4: Verification on Source Validation (R12) Prior to Target Deployment
- Re-run the calibrated gate against the Source validation set (S016–S109 R12).
- Verify that the gate achieves non-zero command yield on active intervals while maintaining empirical false-activation rates $\le \alpha$ on rest intervals.

---

### 4. Safety & Literature Context Reminder

- **Offline Decoding vs. Real-World Operation:** High offline AUC or simulated command replay **does not** prove physical robot safety, arm/gripper reliability, or closed-loop user tolerability.
- **Human-in-the-Loop Dynamics:** In actual closed-loop BCI, total gate suppression induces user frustration, alters user mental strategy, and increases signal non-stationarity.
- All pipeline adjustments, hyperparameter choices, and threshold rules must be finalized strictly on development partitions before prospective testing.
