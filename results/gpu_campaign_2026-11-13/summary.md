# Completed GPU campaign: development readout

The Google Drive campaign completed all six source runs and all 810 target trajectories. The saved `development_summary.json` contains 54/54 complete recipe groups (2 model recipes × 3 target training modes × 3 learning rates × 3 scored epochs). Each group has 45 subject/seed observations: 15 people, three seeds. The campaign scored 2,430 target checkpoints. The notebook results cell was run against the saved Drive summary on November 13.

| Target training mode | Best complete recipe by mean R08 balanced accuracy | R08 balanced accuracy | Raw command during REST | Missed imagery with raw command |
| --- | --- | ---: | ---: | ---: |
| Scratch | EEGNet32, LR .001, epoch 300 | 64.59% | 41.53% | 48.59% |
| Full transfer | EEGNet8, LR .0001, epoch 300 | **68.15%** | 40.11% | 47.95% |
| Classifier only | EEGNet32, LR .0005, epoch 100 | 66.67% | 43.82% | 44.72% |

These rows are the maximum within each training mode across the declared model, learning rate, and epoch choices. They are selected using R08 development results, so the differences are optimistic selection estimates, not independent test results. Seeds do not increase the independent participant count. Both EEGNet sizes and all three modes trained enough to make the earlier 25-update comparison less decisive. More adaptation improved the best scratch recipe, but the best transfer recipe still leads this search on mean accuracy.

The raw command tradeoff is poor across these representative winners. The 68.15% recipe still activates during about 40% of REST time and misses nearly 48% of imagery time. That makes accuracy alone a bad criterion for robot command selection. This campaign did not show that a larger EEGNet or longer training solves command delivery, and it did not run new robot rollouts.

## Recommended decision

Stop the broad model search. Freeze a small development-only command comparison using the best complete transfer recipe and the established CSP/Riemannian baselines. Set a utility rule that reports REST activation, missed imagery, onset delay, and 20-second robot-horizon delivery together. Reject an always-HOLD solution. Carry one selected operating point into the existing robot replay harness with matched always-enable and matched-duration controls, then write up completion and unintended REST motion side by side. Because target R12 has already been inspected, any reuse of it is exploratory. Independent confirmation needs new compatible recordings.

## Reproduction and integrity

- Source: [development_summary.json](development_summary.json) copied from the completed Google Drive campaign on November 13. The result file was produced by `eeg_decoder.gpu_campaign summarize`, which verifies every scored checkpoint hash before reporting a group.
- Original trained code: commit `895845a0a6b600739eff84049e08f56abf0f922d`. The Colab notebook has since been repaired to run the summary on a fresh runtime without a GPU or prior cell state; this does not change the training code or saved checkpoints.
- Drive campaign folder: https://drive.google.com/drive/folders/1SFqoYsksN2MsPckNeIRVDf6wrLIENv4S. Large checkpoints remain on Drive. Local Git contains the compact summary, not all model binaries.
- Target R12 was not used by this campaign. The saved result is an exploratory R08 development comparison, not a prospective risk or robot-safety guarantee.
