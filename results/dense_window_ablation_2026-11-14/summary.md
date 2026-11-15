# Dense causal-window ablation (R08 development)

Seeds: 42, 123, 2026. Participants: 15.
Source pretraining stopped by the frozen 60-epoch patience rule after seed 42: 99 epochs, seed 123: 65 epochs, seed 2026: 121 epochs; every target adaptation ran 300 epochs.
Seeds are averaged within participant; cohort SD uses the 15 participants.
R08 is development data and is not independent confirmation. Target R12 was not loaded.

| Metric | Mean across participants | SD across participants |
|---|---:|---:|
| balanced accuracy | 0.625 | 0.102 |
| raw rest activation | 0.421 | 0.125 |
| raw imagery coverage | 0.594 | 0.117 |
| raw missed imagery | 0.406 | 0.117 |
| raw rest episode activation | 0.942 | 0.060 |
| confidence rest activation | 0.035 | 0.039 |
| confidence imagery coverage | 0.075 | 0.092 |
| confidence missed imagery | 0.925 | 0.092 |
| temporal rest activation | 0.020 | 0.029 |
| temporal imagery coverage | 0.035 | 0.057 |
| temporal missed imagery | 0.965 | 0.057 |

## Historical recipe context

The earlier selected EEGNet8 transfer recipe (LR .0001, epoch 300) reported 68.15% balanced accuracy, 40.11% raw REST activation, and 47.95% missed imagery across 45 participant-seed rows. This run reports 62.52%, 42.07%, and 40.61%, respectively. Dense training modestly reduced missed imagery in this exploratory comparison but reduced balanced accuracy and did not reduce REST activation. The compact historical summary has no paired rows, so these are descriptive differences, not paired inference.
