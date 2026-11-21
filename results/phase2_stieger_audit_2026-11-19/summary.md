# Phase 2 Stieger Corpus Audit & Protocol Definition (WP1 & WP2)

## 1. Pre-Inspection Deterministic Split & Cohort Inventory

Metadata for all `599` files (`598` session `.mat` recordings across `62` subjects + `README.pdf`, `351.01 GiB` total) was fetched from Figshare article `13123148` (`doi:10.6084/m9.figshare.13123148.v1`) and stored in [stieger_figshare_manifest.json](stieger_figshare_manifest.json). **Before downloading or inspecting any `.mat` signal**, participants were deterministically partitioned (`seed = 20261119`) in [configs/phase2_stieger_split.json](../../configs/phase2_stieger_split.json):

| Partition / Nested Scale | Subjects | Sessions | Raw MAT Size (GiB) | Projected 100 Hz 9-ch Shards (GiB) | Purpose & Access Policy |
|---|---:|---:|---:|---:|---|
| **`source_train` (`scale_10pct`)** | 4 | 28 | 15.677 | 0.415 | Nested source pretraining prefix (~9.1% of source subjects) |
| **`source_train` (`scale_30pct`)** | 13 | 111 | 63.323 | 1.646 | Nested source pretraining prefix (~29.5% of source subjects) |
| **`source_train` (`scale_100pct`)** | 44 | 416 | 243.068 | 6.168 | Full source pretraining pool (`S_order` fixed in split config) |
| **`dev`** (`S1`, `S2` + 6 sampled) | 8 | 84 | 50.425 | — | Data audit, architecture/window selection, hyperparams |
| **`test`** (reserved, unopened) | 10 | 98 | 57.517 | — | Untouched until WP4 locked evaluation (earliest session cal, later eval) |
| **Total corpus** | **62** | **598** | **351.010** | **8.867** | ~`572.6` unique deduplicated hours (`57.45 min`/session mean across 3 audited) |

## 2. Three-Session End-to-End Structural & Continuity Audit

Audited `S1_Session_1` (baseline day `20170116`), `S1_Session_7` (`20170417`), and `S2_Session_1` (`20170126`) sequentially via [analysis/audit_stieger_phase2.py](../../analysis/audit_stieger_phase2.py). Each raw `.mat` (~`623.16 MB` mean) matched its exact Figshare byte count and MD5, converted causally to a compact `100 Hz` `9`-channel `.npz` shard (`15.30–16.56 MB`, `0.0255x` raw size), and was deleted immediately. Exact per-session metrics are recorded in [audit_metrics.json](audit_metrics.json).

- **Montage & signal integrity:** All three files are MATLAB 5.0 structs sampled at `1000 Hz` across `62` channels (`CZ` normalized cleanly to canonical `Cz`). All `9` project channels (`C3, Cz, C4, C1, C2, FC1, FC2, CP1, CP2`, 1-based indices `[26, 28, 30, 27, 29, 18, 20, 36, 38]`) had `0` non-finite samples and `0` overlap with `noisechan` (`['6']` in `S1_Session_1`, `['41']` in `S1_Session_7`, `[]` in `S2_Session_1`).
- **Trial counts & exact schedule:** Every session has `450` trials (`150 LR`, `150 UD`, `150 2D`; `0` artifact trials in these 3 sessions) organized into `18` runs × `25` trials (`432` adjacent same-run pairs). Every trial begins cleanly at `t = -2000 ms` (`2 s` blank inter-trial interval), `t = 0 ms` (`2 s` visual target cue), and `t = +2000 ms` (closed-loop feedback lasting `0.12–6.04 s` across `126–134` distinct `resultind` lengths), followed by `1.0 s` `post_trial_outcome`. (Our strict raw `positionx/py` NaN-mask check flagged `450/450` trials, so cursor traces are excluded from modeling features pending separate inspection if ever needed.)
- **1.0 s within-run overlap & discontinuities:** Across all `432` same-run adjacent pairs per session, the final `1.0 s` of trial `k` (`post_trial_outcome`) matches the first `1.0 s` (`[-2000, -1000) ms`) of trial `k+1` at median 100 Hz CAR correlation **`0.9987–0.9990`** (`0.9845–0.9885` on un-referenced raw `C3/Cz/C4`), while cell boundaries exhibit `2.24–2.86 uV` DC shifts (`edge jump = 9.61–10.84x` median 1-sample step). Filtering is therefore reset independently per trial using causal step-matched `sosfilt_zi` (`eeg_decoder/stieger.py`), and the overlapped `[-2000, -1000) ms` prefix is excluded from `clean_intertrial_idle`.

## 3. Empirical Visual-Feedback Confound & Choice of Pretraining Labels

Chronological 5-fold within-session OAS Riemannian+LDA evaluations on the 100 Hz shards (causal 8–30 Hz `sosfilt` with step-matched `zi`) expose a critical structural confound if deliberate relaxation (`targetnumber == 4`, cursor Down) or visual feedback vs. blank idle are conflated:

| Session (Window) | Active MI vs. Clean Idle (`[-1s, 0)`) | Deliberate Relax (`Down`) vs. Clean Idle | Cue MI (`[0, 2s)`) vs. Clean Idle | Feedback Active MI vs. Deliberate Relax (`Down`) |
|---|---:|---:|---:|---:|
| **`S1_Session_1`** (`2.0 s` / `1.0 s`) | `0.8625` / `0.9066` | `0.7962` / `0.8395` | `0.6164` / `0.6606` | `0.4814` / `0.4675` |
| **`S1_Session_7`** (`2.0 s` / `1.0 s`) | `0.9419` / `0.9469` | `0.8560` / `0.8448` | `0.5888` / `0.7181` | **`0.8944`** / **`0.8248`** |
| **`S2_Session_1`** (`2.0 s` / `1.0 s`) | `0.8022` / `0.8300` | `0.7915` / `0.8320` | `0.6647` / `0.6752` | `0.4066` / `0.4151` |

- Both `feedback_active_mi` and `feedback_deliberate_relax` separate strongly from blank `clean_intertrial_idle` (`ROC-AUC 0.79–0.95`) even when `MI vs. Relax` is at chance on Day 1 (`0.41–0.48`), reflecting visual/task-engagement changes once cursor feedback begins.
- **Selected pretraining label scheme ([configs/phase2_training.json](../../configs/phase2_training.json)):** Preserve all `7` sample states (`0..6`) in every shard, and pretrain `Phase2EEGNet` (`EEGNet8` causal wrapper) on a **3-class objective** (`0: clean_intertrial_idle [t in [-1000, 0) ms]`, `1: feedback_deliberate_relax`, `2: feedback_active_mi`) with inverse-sqrt class weights. This keeps deliberate relaxation explicitly separated from true idle and tracks `val_auc_mi_vs_clean_idle` and `val_auc_mi_vs_deliberate_relax` independently.

## 4. Evaluation Endpoints & Resumable Pipeline Verification

- **Valid Stieger endpoint:** Held-out `test_subjects` later sessions (after adapting on each participant's earliest session) validly test **cross-day causal SMR decoding & MI-vs-relaxation transfer**.
- **Separate continuous asynchronous endpoint:** Because Stieger stores overlapping trial segments with fixed `2 s` idle + `2 s` visual cue and user-dependent trial end times (`0.12–6.04 s`), Stieger alone cannot test continuous un-cued request timing. Independent continuous benchmarks (shortlist: `Lee2019_OpenBMI`, `Kaya2018_ContinuousMI`, `Ofner2017_UpperLimb`, `BCICIII_IVc`) are designated separately before WP4 robot replay; Berlin `BCICIV-1` remains restricted to dev/historical comparison.
- **Verified local CPU smoke + interrupt/resume test ([cpu_smoke_test.json](cpu_smoke_test.json)):** Running [eeg_decoder/phase2_train.py](../../eeg_decoder/phase2_train.py) on `S1_Session_1 + S1_Session_7` train (`15,629` windows) and `S2_Session_1` Run-1-EA val (`8,253` windows, `1.0 s`) stopped cleanly at `step=10`, verified Drive SHA-256 (`drive_verified: true`), and resumed to `step=20` (`val_loss: 1.1050 -> 1.1010`, `auc_mi_idle: 0.6184 -> 0.6882`, `747.2 win/s`).

## 5. Next Executable Ingestion & Colab GPU Profiling Commands

1. Open [notebooks/FRI_BCI_Phase2_EEG_Scaling.ipynb](../../notebooks/FRI_BCI_Phase2_EEG_Scaling.ipynb) in Google Colab (H100 preferred), run **Cells 2–4** to stage the 3 audited dev shards onto `/content/phase2_shards` and profile GPU interrupt/resume throughput.
2. CLI equivalent on Colab/CPU for staging the `scale_10pct` (`28` source + `84` dev shards):

```bash
python -u -m eeg_decoder.phase2_train ingest \
  --scale scale_10pct --include-dev \
  --shard-dir /content/phase2_shards \
  --raw-dir /content/phase2_raw_tmp \
  --drive-shard-dir /content/fri_drive/MyDrive/FRI_BCI/phase2_stieger_scaling/shards_100hz_9ch
```
