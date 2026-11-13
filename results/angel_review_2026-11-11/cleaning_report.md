# Fixed offline cleaning experiment

Provisional frontal-proxy ICA exclusions require visual review. No dedicated EOG. AutoReject is experimental, not established optimal. ICA and both AutoReject stages fit runs 4/8 only. No CSP hyperparameter search. Run 12 is exploratory. Offline filtering uses future samples and can smear transients across annotations.

| Model | Arm | Accepted BA | Test coverage | BA on shared accepted windows | Coverage-adjusted correct fraction |
|---|---|---:|---:|---:|---:|
| pooled | causal | 49.3% | 100.0% | 49.6% | 49.3% |
| pooled | zero_phase | 50.0% | 100.0% | 50.3% | 50.0% |
| pooled | zero_phase_lw | 50.7% | 100.0% | 51.0% | 50.7% |
| pooled | clean_lw | 52.3% | 99.3% | 52.3% | 52.0% |
| subject_001 | causal | 76.7% | 100.0% | 76.7% | 76.7% |
| subject_001 | zero_phase | 70.0% | 100.0% | 70.0% | 70.0% |
| subject_001 | zero_phase_lw | 73.3% | 100.0% | 73.3% | 73.3% |
| subject_001 | clean_lw | 73.3% | 100.0% | 73.3% | 73.3% |
| subject_002 | causal | 73.3% | 100.0% | 73.3% | 73.3% |
| subject_002 | zero_phase | 70.0% | 100.0% | 70.0% | 70.0% |
| subject_002 | zero_phase_lw | 73.3% | 100.0% | 73.3% | 73.3% |
| subject_002 | clean_lw | 73.3% | 100.0% | 73.3% | 73.3% |
| subject_003 | causal | 53.3% | 100.0% | 53.3% | 53.3% |
| subject_003 | zero_phase | 50.0% | 100.0% | 49.8% | 50.0% |
| subject_003 | zero_phase_lw | 53.3% | 100.0% | 53.3% | 53.3% |
| subject_003 | clean_lw | 53.1% | 96.7% | 53.1% | 50.0% |
| subject_004 | causal | 93.3% | 100.0% | 93.3% | 93.3% |
| subject_004 | zero_phase | 93.3% | 100.0% | 93.3% | 93.3% |
| subject_004 | zero_phase_lw | 90.0% | 100.0% | 90.0% | 90.0% |
| subject_004 | clean_lw | 90.0% | 100.0% | 90.0% | 90.0% |
| subject_005 | causal | 56.7% | 100.0% | 56.7% | 56.7% |
| subject_005 | zero_phase | 60.0% | 100.0% | 60.0% | 60.0% |
| subject_005 | zero_phase_lw | 53.3% | 100.0% | 53.3% | 53.3% |
| subject_005 | clean_lw | 60.0% | 100.0% | 60.0% | 60.0% |

Rejected windows are abstentions; coverage-adjusted correct fraction counts these as unsuccessful trials. Shared accepted-window scores compare the controls with cleaned data on identical test IDs, while their training sets may differ. ICA/spline repair/full-cap reference effects are combined in the full arm. Ledoit-Wolf shrinks covariance, not sample amplitudes. See docs/offline_cleaning.md for protocol and sources.

## Retention and ICA review

| Participant | Train rest kept | Train imagery kept | Test rest kept | Test imagery kept | Provisional ICA exclusions |
|---|---:|---:|---:|---:|---|
| subject_001 | 30/30 | 30/30 | 15/15 | 15/15 | [] |
| subject_002 | 30/30 | 30/30 | 15/15 | 15/15 | [] |
| subject_003 | 30/30 | 29/30 | 15/15 | 14/15 | [] |
| subject_004 | 30/30 | 30/30 | 15/15 | 15/15 | [] |
| subject_005 | 30/30 | 30/30 | 15/15 | 15/15 | [0] |

## Cleaning settings

Voltage threshold: 150 microvolts peak-to-peak (fixed when numeric), at both pre-ICA and post-ICA stages.
ICA fitting band: [1, 40] Hz; the original experiment already used the 1 Hz high-pass.
Repair candidates: [1, 2, 4, 8, 12, 16, 19]; consensus candidates: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].

A repair limit does not cap the number of flagged channels in a retained window. Standard AutoReject may leave some flagged channels unrepaired when the consensus rejection threshold is not reached.

| Participant | Pre-ICA repair / consensus | Post-ICA repair / consensus | Test windows kept with unrepaired flags | Maximum repaired channels per kept test window |
|---|---|---|---:|---:|
| subject_001 | 8 / 0.5 | 2 / 0.2 | 1 | 2 |
| subject_002 | 8 / 0.2 | 1 / 0.1 | 0 | 1 |
| subject_003 | 12 / 0.6 | 4 / 0.1 | 5 | 4 |
| subject_004 | 8 / 0.3 | 2 / 0.1 | 0 | 1 |
| subject_005 | 12 / 0.3 | 1 / 0.1 | 0 | 1 |
