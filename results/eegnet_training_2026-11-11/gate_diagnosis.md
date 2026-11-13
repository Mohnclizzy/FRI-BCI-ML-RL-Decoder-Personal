# Development-only gate diagnosis - 2026-11-11

This diagnostic replays only target R08 with the frozen models and thresholds. It does not select parameters, refit models, or reopen R12. R08 is calibration data, so these measurements are descriptive and not held-out validation.

## Findings

- All 30 models already produce zero accepted commands on R08. The failure is visible before target test evaluation.
- In 29/30 models, the maximum score never reaches the activation threshold (calibrated threshold + 0.05). Transfer S001 has one above-activation sample, insufficient for 0.5-second persistence at a 0.25-second stride. Three consecutive qualifying samples are required from a fresh HOLD state.
- With 15 REST episodes and alpha=0.1, (errors+1)/16 <= 0.1 requires zero observed episode exceedances. The smallest passing fixed-grid threshold must exceed every observed REST episode maximum.
- The gate then adds a 0.05 activation margin and persistence on top of that conservative threshold. These additional constraints explain the observed rejection; no runtime or timestamp rejection is needed to explain it.
- Scratch scores are concentrated near 0.5. A fixed margin of 0.05 exceeds the observed useful score range in this run. This is not proof that lowering the margin would produce reliable commands.

## Per-model evidence

| Model | Subject | Hold threshold | Enable threshold | Maximum R08 score | Above-enable samples | Longest streak |
|---|---|---:|---:|---:|---:|---:|
| transfer | S001 | 0.810 | 0.860 | 0.8672 | 1 | 1 |
| transfer | S002 | 0.830 | 0.880 | 0.8293 | 0 | 0 |
| transfer | S003 | 0.695 | 0.745 | 0.6906 | 0 | 0 |
| transfer | S004 | 0.765 | 0.815 | 0.7797 | 0 | 0 |
| transfer | S005 | 0.770 | 0.820 | 0.8068 | 0 | 0 |
| transfer | S006 | 0.700 | 0.750 | 0.6963 | 0 | 0 |
| transfer | S007 | 0.810 | 0.860 | 0.8073 | 0 | 0 |
| transfer | S008 | 0.690 | 0.740 | 0.7286 | 0 | 0 |
| transfer | S009 | 0.755 | 0.805 | 0.7641 | 0 | 0 |
| transfer | S010 | 0.720 | 0.770 | 0.7172 | 0 | 0 |
| transfer | S011 | 0.685 | 0.735 | 0.7008 | 0 | 0 |
| transfer | S012 | 0.755 | 0.805 | 0.7504 | 0 | 0 |
| transfer | S013 | 0.850 | 0.900 | 0.8464 | 0 | 0 |
| transfer | S014 | 0.790 | 0.840 | 0.7940 | 0 | 0 |
| transfer | S015 | 0.870 | 0.920 | 0.8675 | 0 | 0 |
| scratch | S001 | 0.515 | 0.565 | 0.5142 | 0 | 0 |
| scratch | S002 | 0.530 | 0.580 | 0.5265 | 0 | 0 |
| scratch | S003 | 0.525 | 0.575 | 0.5229 | 0 | 0 |
| scratch | S004 | 0.540 | 0.590 | 0.5514 | 0 | 0 |
| scratch | S005 | 0.510 | 0.560 | 0.5078 | 0 | 0 |
| scratch | S006 | 0.530 | 0.580 | 0.5323 | 0 | 0 |
| scratch | S007 | 0.540 | 0.590 | 0.5451 | 0 | 0 |
| scratch | S008 | 0.505 | 0.555 | 0.5022 | 0 | 0 |
| scratch | S009 | 0.535 | 0.585 | 0.5342 | 0 | 0 |
| scratch | S010 | 0.520 | 0.570 | 0.5211 | 0 | 0 |
| scratch | S011 | 0.535 | 0.585 | 0.5367 | 0 | 0 |
| scratch | S012 | 0.535 | 0.585 | 0.5332 | 0 | 0 |
| scratch | S013 | 0.520 | 0.570 | 0.5189 | 0 | 0 |
| scratch | S014 | 0.540 | 0.590 | 0.5354 | 0 | 0 |
| scratch | S015 | 0.540 | 0.590 | 0.5366 | 0 | 0 |

## Next experiment proposal, not executed

1. Preserve all existing models, thresholds, R12 outputs and checksums. Do not tune or choose gates from their R12 performance.
2. Before model fitting, split source subjects into disjoint model-training and gate-development groups. Use those development subjects to study score distributions, onset persistence and REST-versus-imagery command sensitivity. Keep complete subjects and recordings together; do not randomly split overlapping windows.
3. Predeclare a small set of gate candidates and report both REST activation and imagery delivery. A gate that never activates must explicitly fail a usability check rather than appear successful solely because REST activation is zero. Keep any candidate-selection data distinct from later calibration/evaluation data.
4. Refit/freeze the complete model and gate procedure before evaluating on genuinely new, independently collected compatible recordings. Existing target R12 may support clearly labeled exploratory comparisons, but cannot regain fresh-confirmation status.
5. Treat any risk statement as empirical unless exchangeability and the complete calibration procedure justify more. Do not translate command statistics into physical robot safety or success.

No gate code, model, threshold or scientific configuration was changed by this diagnosis.

## Antigravity coordination and review adjudication

Antigravity CLI 1.5.6 completed a bounded review of supplied gate code and R08 diagnostics. The initial tool-based attempt was denied by headless permissions; the successful retry required no shell permission expansion. Its raw advisory output is saved in antigravity_review.md.

Accepted: the zero-exceedance arithmetic, activation margin and persistence explain the observed command suppression. The independent measurements above support those mechanisms.

Not accepted as established findings or changes:

- Near-0.5 scratch scores do not establish underfitting, class imbalance, or a specific representation defect. Those need separate training/development diagnostics.
- Score-exceedance calibration is an intentional conservative upper envelope, not automatically an implementation bug or an invalid loss. Its conservatism should be measured against command utility.
- Splitting REST episodes to inflate the count does not create independent/exchangeable samples or preserve the original episode-level risk unit. Do not use that to manufacture a stronger bound.
- Calibrating actual gate loss is only a research proposal. Count any active command overlapping REST, including commands carried from preceding imagery, not merely transitions inside REST. Validate relevant monotonicity and statistical assumptions before applying a CRC formula.
- Temperature scaling cannot guarantee calibrated posteriors or prevent high REST scores, and using the same source validation data for tuning and confirmation would not provide independent evidence.
- Neither the review nor this diagnostic certifies absence of all implementation bugs. No proposed algorithm change was applied.
