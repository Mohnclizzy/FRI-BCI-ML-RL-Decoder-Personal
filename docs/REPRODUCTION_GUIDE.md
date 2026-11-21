# Reproduction Guide (CPU Clean-Checkout Routes)

This guide documents two verified routes to reproduce the headline findings in [report/final/final_report.pdf](../report/final/final_report.pdf) from a fresh clone on branch `codex/robot-learning-design`, without touching frozen evidence folders (`results/bounded_eeg_robot_2026-11-14/`, `results/dense_window_ablation_2026-11-14/`, `results/bciciv_ds1_request_pilot_2026-11-19/`) or any author-local paths.

---

## Dependencies & Environment Setup (Windows or Linux CPU)

Tested Python environment: `Python 3.14` (compatible with `Python >= 3.11`; `requirements-lock.txt` + `requirements-riemannian.txt` pin exact tested Windows versions). Notice **`pyriemann==0.12`** (from `requirements-riemannian.txt`) is required for Route B (`analysis/bciciv_ds1_request_pilot.py`), whereas Route A requires only base packages in `requirements.txt` + `pyriemann` if running the full pytest suite.

> **Windows line-ending note:** Several committed result manifests hash exact stored bytes (`results/bciciv_ds1_request_pilot_2026-11-19/SHA256SUMS.json`, `results/robomimic_latch6_zoffset_2026-11-04/latch6_rollout.py`), and `results/` contains files indexed with both `LF` and `CRLF`. The root `.gitattributes` sets `results/** -text` so Git checks out exact stored bytes on every platform without EOL conversion. As defensive hygiene on Windows systems with `core.autocrlf=true`, cloning with `-c core.autocrlf=false` is also safe and recommended.

```powershell
git clone -c core.autocrlf=false --branch codex/robot-learning-design https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal.git
cd FRI-BCI-ML-RL-Decoder-Personal

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-riemannian.txt
```

On Linux / macOS, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

---

## Route A — Short Route (Committed Records, CPU, ~15 s)

Rebuilds the Study~2 overview figure and verified JSON summary tables directly from committed records after checking all SHA-256 manifests, reproduces the PhysioNet `R04 == R08` clock audit from committed annotations, and runs the full test suite into a fresh output directory.

```powershell
# 1. Verify committed SHA-256 manifests and regenerate Study 2 figure + report tables
.\.venv\Scripts\python.exe -m analysis.build_final_report_artifacts --out-dir outputs/repro_short

# 2. Verify PhysioNet R04 == R08 schedule identity and 4.1 s clock baseline
.\.venv\Scripts\python.exe analysis/audit_cue_clock.py

# 3. Run unit and artifact verification tests
.\.venv\Scripts\python.exe -m pytest -q
```

### Expected Route A outputs

- `outputs/repro_short/bciciv_ds1_pilot_overview.png`: Study~2 overview figure (4 human participants $\times$ 3 controllers, 720$\to$255$\to$161$\to$154.0 imagery funnel, pooled request-time outcomes + exploratory slot rotation).
- `outputs/repro_short/verified_report_tables.json`:
  - Manifest verification: `study2_manifest_files_verified = 6`, `study1_core_files_verified = 7`.
  - Study~1 (`PhysioNet R08`): `EEGNet` cue BA `68.15%`, `selected` `42/45` retained (`0.175 m` REST EEF path) vs.\ `raw` `45/45` (`0.209 m`).
  - Engineering audit: `15/15` identical R04/R08 schedules (`7,278/7,278` endpoints = `100%`), `4.1 s` timer `96.08%`, `90/225` (`40.0%`) 20 s slot ceiling.
  - Study~2 (`BCICIV-1` Runs 2--4): `any_240_post_acquisition_loss = false`, accumulator **`161`** imagery / **`62`** REST / **`59`** transition accepted starts $\to$ **`154.0 / 59.0 / 57.0`** continuously retained lifts (3-seed mean); threshold `139 / 60 / 56` (`129.0 / 57.7 / 52.0`); timer `130 / 133 / 70` (`123.0 / 130.0 / 68.0`).
- `analysis/audit_cue_clock.py` stdout/JSON confirms `15/15` participants share identical intervals (`7,278/7,278`).

---

## Route B — Full Route (Official Archives + Frozen EEG Refit + Native Robot Grid)

Refits all four human 3-class Riemannian Tangent-Space + Shrinkage LDA decoders on Evaluation Run~1 from scratch using official public EEG archives, recomputes the BCICIV-1 clock-only baselines, and joins the resulting timestamped request streams with verified 240-rollout `robomimic-latch6-v2.0.0` physics outcomes into a fresh output directory.

### Step B1: Download & verify official BCI Competition IV Dataset 1 archives

Place both official zip archives in `<ARCHIVE_DIR>` (e.g., `outputs/bciciv_archives`):

- **100 Hz continuous EEG signals (`BCICIV_1_mat.zip`):**
  `https://www.bbci.de/competition/download/competition_iv/BCICIV_1_mat.zip`
  Expected SHA-256: `f7acc9420158e6d58629589040ced64b49ffd8e946c4024f85cc77896acdf686`
- **1000 Hz official evaluation labels (`true_labels.zip`):**
  `https://www.bbci.de/competition/iv/results/ds1/true_labels.zip`
  Expected SHA-256: `b34de8373d6476ff534f3a7496397ed86726c04dd12f610d84b8baaa71e28c72`

### Step B2: Prepare verified 240-rollout native robot summaries (CPU fast check vs.\ Linux/WSL2 regeneration)

All 240 `(seed in 0..2, window_start in 1,6,..,396)` native `robomimic-latch6-v2.0.0` rollouts are simulator-deterministic across start steps and independent of EEG inputs:

- **Option B2a — Reuse committed versioned physics bundle (Windows or Linux CPU, <1 s):**
  Verify `results/bciciv_ds1_request_pilot_2026-11-19/SHA256SUMS.json`, confirm `robomimic_240_rollouts_bundle.tar.gz` (`SHA-256: 39a8035775c2e66e4f68162d8a9263bda818f97a6d44f3b3a73c295aae24fa5e`) contains all 240 `summary.json` + `telemetry.jsonl` traces with 0 post-acquisition drops, and copy `robomimic_240_summaries.json` into your fresh output directory:

  ```powershell
  New-Item -ItemType Directory -Force -Path outputs/repro_full | Out-Null
  Copy-Item results/bciciv_ds1_request_pilot_2026-11-19/robomimic_240_summaries.json outputs/repro_full/
  ```

- **Option B2b — Full native MuJoCo regeneration (Linux / WSL2 only, ~3 min on 12 CPU workers):**
  *Platform requirement:* `mujoco-py==2.1.2.14` + `robosuite 1.2` require Linux (`libosmesa6`, `libosmesa6-dev`, `patchelf`, MuJoCo 2.1 binaries in `~/.mujoco/mujoco210`) and cannot build natively inside Windows Python 3.14. Pinned WSL2/Linux environment used for committed rollouts:
  `Python 3.8.20`, `torch==1.13.1+cpu`, `torchvision==0.14.1+cpu`, `mujoco-py==2.1.2.14`, `gym==0.23.1`, `robosuite@d9baa17d`, `robomimic@2804dd97`, headless OSMesa (`MUJOCO_GL=osmesa`).

  ```bash
  python simulation/scripts/run_one_request_window_grid.py \
    --ckpt artifacts/lift_ph_low_dim_epoch_1000_succ_100.pth \
    --out-dir outputs/repro_full \
    --workers 12
  ```

### Step B3: Run clock audit & frozen EEG decoder refit on CPU (~15 s)

```powershell
# Recompute BCICIV-1 clock baselines & cue durations from official true_labels.zip
.\.venv\Scripts\python.exe -m analysis.audit_bciciv_ds1_clock `
  --labels-zip <ARCHIVE_DIR>\true_labels.zip `
  --out outputs/repro_full/bciciv_ds1_clock_audit.json

# Refit Run 1 decoders, freeze controllers, score Runs 2-4, join with 240 robot rollouts
.\.venv\Scripts\python.exe -m analysis.bciciv_ds1_request_pilot `
  --data-dir <ARCHIVE_DIR> `
  --out-dir outputs/repro_full
```

### Expected Route B outputs

- `outputs/repro_full/bciciv_ds1_clock_audit.json`: human mean continuous-cue Run~1$\to$Runs~2--4 replay = `0.5063`, fixed `4.1 s` timer = `0.5001`, imagery duration `[1.502, 8.358]` s, REST `[1.506, 7.998]` s, `ds1c_run1_vs_run3_exact_match = true`.
- `outputs/repro_full/pilot_metrics.json` & `exported_timestamped_requests_runs234.json`: exact parity with committed `results/bciciv_ds1_request_pilot_2026-11-19/` (`469` emitted / `255` imagery detected / `161` accepted / `154.0` retained imagery, `62` / `59.0` REST, `59` / `57.0` transition for accumulator; `114.28 / 108.33 / 59.39` slot rotation; `0` post-acquisition drops).

### Optional: Recompile the PDF report

From `report/final/` using Tectonic (`0.17.0`+) or standard XeLaTeX/pdfLaTeX + BibTeX (`tectonic final_report.tex`).

### Study 1 recovery pointer

Study~1 (`PhysioNet`, 15 participants, 270 rollouts) large campaign checkpoints remain on Drive with full SHA-256 verification instructions in [BOUNDED_STUDY_HANDOFF_2026-11-14.md](BOUNDED_STUDY_HANDOFF_2026-11-14.md). Committed Study~1 compact metrics (`groups.json`, `rows.json`, `robot/robot_results/aggregate.json`) and representative video (`results/bounded_eeg_robot_2026-11-14/robot/robot_results/S001_seed0_selected/video.mp4`) are verified directly in Route~A.

---

## Verified Clean-Checkout Log

Executed on `2026-11-19` on Windows (`Python 3.14`, plain `git clone --branch codex/robot-learning-design` without overriding global `core.autocrlf=true`, isolated temporary clone outside author working tree and scratch directories):

- **Route A (`22.59 s` total):**
  - `python -m analysis.build_final_report_artifacts --out-dir <TEMP_SHORT>` exited `0`, verified `6` Study~2 manifest hashes and `7` core Study~1 manifest hashes, regenerated `bciciv_ds1_pilot_overview.png`, and produced `verified_report_tables.json` matching the committed file (`exact JSON equality = True`).
  - `python analysis/audit_cue_clock.py` exited `0` (`15/15` identical R04/R08 schedules, `7,278/7,278` endpoints = `100%`, `4.1 s` timer `96.08%`).
  - `python -m pytest -q` passed **`123 passed, 2 warnings, 8 subtests passed`** in `16.74 s` (including `test_historical_phase6_latch6_rollout_hash_preserved` and `test_build_final_report_artifacts_from_committed_records`, confirming `.gitattributes` preserves mixed `LF`/`CRLF` byte hashes across plain Windows checkouts).
- **Route B (`65.11 s` total + physics archive check):**
  - Inspected committed `results/bciciv_ds1_request_pilot_2026-11-19/robomimic_240_rollouts_bundle.tar.gz`: verified `240` `summary.json` and `240` `telemetry.jsonl` files (`480` total members) with `0` post-acquisition drops, and reused `robomimic_240_summaries.json` in a fresh output directory.
  - `python -m analysis.audit_bciciv_ds1_clock` on verified `true_labels.zip` matched committed `report/final/bciciv_ds1_clock_audit.json` (`0.5063` Run~1$\to$Runs~2--4 replay, `0.5001` `4.1 s` timer, `ds1c` Run~3 `1.0000` match; `exact JSON equality = True`).
  - `python -m analysis.bciciv_ds1_request_pilot` refitted Run~1 decoders and re-scored Runs~2--4 on CPU with **100% JSON equality** against committed `pilot_metrics.json` and `exported_timestamped_requests_runs234.json` (`161/62/59` accepted and `154.0/59.0/57.0` retained for accumulator; `139/60/56` / `129.0/57.7/52.0` for threshold; `130/133/70` / `123.0/130.0/68.0` for timer; slot rotation `114.28/108.33/59.39`).
- **Platform notes recorded:**
  - Windows `core.autocrlf=true` rewrites `LF`-indexed text files unless `.gitattributes` sets `results/** -text`; adding `.gitattributes` resolved clean-clone SHA-256 verification without changing frozen `results/` files.
  - Native MuJoCo `robomimic` grid regeneration (`simulation/scripts/run_one_request_window_grid.py`) requires Linux/WSL2 (`Python 3.8.20`, `mujoco-py 2.1.2.14`, `robosuite@d9baa17d`, `robomimic@2804dd97`, `libosmesa6`); Windows CPU verification reuses the committed versioned 240-rollout physics telemetry and bundle.
