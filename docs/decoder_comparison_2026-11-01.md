# Decoder comparison (2026-11-01)

## Scope and protocol

This report records the existing comparison artifacts for subjects 1–5. The configured evaluation is `held_out_run_same_subjects`: runs 4 and 8 train the models and run 12 is the held-out test recording for each of the same five subjects. Each subject therefore has 60 training windows and 30 test windows; the aggregate scores pool 150 test windows. “Pooled” uses a single model trained across subjects; “individual” fits one model per subject using that subject’s runs 4 and 8. The metric `rest_false_activation_fraction` is the fraction of true REST windows predicted as HAND_IMAGERY.

The CSP/LDA and Riemannian tangent-space/LDA results below are copied from `outputs/subject_comparison/metrics.json` and `outputs/subject_comparison_riemannian/metrics.json`. Values retain the precision in those files.

## Aggregate test results

| Pipeline | Model | Windows | Balanced accuracy | Macro F1 | REST false activation fraction |
|---|---|---:|---:|---:|---:|
| Pooled | CSP + LDA | 150 | 0.49333333333333335 | 0.4900697799248524 | 0.4266666666666667 |
| Individual | CSP + LDA | 150 | 0.7066666666666667 | 0.7066666666666667 | 0.29333333333333333 |
| Pooled | Riemannian covariance/tangent space + LDA | 150 | 0.7 | 0.6977296135417133 | 0.21333333333333335 |
| Individual | Riemannian covariance/tangent space + LDA | 150 | 0.7266666666666667 | 0.7260701082357133 | 0.22666666666666666 |

## Per-subject test results

| Subject | CSP pooled BA | CSP pooled macro F1 | CSP pooled REST false activation | CSP individual BA | CSP individual macro F1 | CSP individual REST false activation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5333333333333333 | 0.4034090909090909 | 0.0 | 0.7666666666666666 | 0.7664071190211346 | 0.2 |
| 2 | 0.4666666666666667 | 0.3181818181818182 | 0.06666666666666667 | 0.7333333333333334 | 0.7222222222222222 | 0.06666666666666667 |
| 3 | 0.36666666666666664 | 0.2682926829268293 | 0.26666666666666666 | 0.5333333333333333 | 0.4034090909090909 | 0.9333333333333333 |
| 4 | 0.5333333333333333 | 0.4034090909090909 | 0.9333333333333333 | 0.9333333333333333 | 0.9330357142857143 | 0.13333333333333333 |
| 5 | 0.5666666666666667 | 0.466484268125855 | 0.8666666666666667 | 0.5666666666666667 | 0.5238095238095237 | 0.13333333333333333 |

| Subject | Riemannian pooled BA | Riemannian pooled macro F1 | Riemannian pooled REST false activation | Riemannian individual BA | Riemannian individual macro F1 | Riemannian individual REST false activation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7666666666666666 | 0.76 | 0.06666666666666667 | 0.6333333333333333 | 0.6329254727474972 | 0.3333333333333333 |
| 2 | 0.6 | 0.5927601809954751 | 0.26666666666666666 | 0.7333333333333334 | 0.7321428571428572 | 0.2 |
| 3 | 0.5666666666666667 | 0.5238095238095237 | 0.13333333333333333 | 0.7666666666666666 | 0.7643097643097643 | 0.3333333333333333 |
| 4 | 0.8333333333333334 | 0.8331479421579533 | 0.2 | 0.8666666666666667 | 0.8666666666666667 | 0.13333333333333333 |
| 5 | 0.7333333333333334 | 0.7285067873303167 | 0.4 | 0.6333333333333333 | 0.6122209165687427 | 0.13333333333333333 |

## Reproduction

From the repository root, with the project’s dependencies installed and the PhysioNet EEGBCI data available at the configured `data/physionet` location, prepare windows and run each comparator into fresh output directories (the comparator requires the destination not to exist):

```powershell
python -m eeg_decoder prepare --config configs/baseline.json
python -m eeg_decoder.compare_subjects --config configs/baseline.json --output outputs/subject_comparison_reproduced --family csp
```

For the Riemannian model, install its pinned additions from `requirements-riemannian.txt` (which includes `requirements-lock.txt`), then run:

```powershell
python -m pip install -r requirements-riemannian.txt
python -m eeg_decoder.compare_subjects --config configs/baseline.json --output outputs/subject_comparison_riemannian_reproduced --family riemannian
```

The checked-in metrics were produced against prepared windows in `outputs/baseline`, using the config recorded in each metrics file. The commands above regenerate the comparison artifacts; exact metric reproduction also depends on using the same data files and software environment.

## Limitations

- These subjects 1–5 / run 12 results were previously inspected. This is an exploratory same-subject held-out-run diagnostic, not fresh independent confirmation. The test recording is separated by run from training, but the people are the same; this does not establish transfer to unseen subjects.
- The per-subject models use that same person’s runs 4 and 8 as calibration data. The metrics explicitly mark probabilities as uncalibrated; probability values must not be treated as reliable confidence estimates.
- The labels represent the dataset’s REST versus HAND_IMAGERY task classes. These offline window scores do not measure command stability, closed-loop feedback, or robot/robotic-hand performance.
- The literature review likewise frames public-data replay as offline decoding evidence, not evidence of live-user adaptation, clinical benefit, or physical hand safety (`docs/literature_reference_summary.md`).
# Fresh-person confirmation after frozen model selection

The follow-up command `python -m eeg_decoder.validate_decoder --output outputs/decoder_validation_001`
fit development subjects 1–5 on run 4 and validated on run 8, excluding run 12.
Individual CSP and Riemannian models both scored 65.33% balanced accuracy.
The predefined tie rule selected CSP. The selection and timestamp were written
before fetching/evaluating subjects 6–10.

On those five confirmation participants, each calibrated using their own runs
4/8 and tested on run 12, individual CSP achieved 59.33% balanced accuracy;
pooled CSP achieved 57.33%. Individual rest-window false activations were 29.33%,
versus 58.67% pooled. Individual balanced accuracies were 53.33%, 66.67%, 73.33%,
46.67%, and 56.67% for subjects 6–10 respectively (30 test windows per person).
The earlier exploratory gain therefore does not justify assuming consistently
strong decoding. This small confirmation cohort is now inspected and must not
be reused as fresh evidence after further tuning. It measures calibrated
within-person decoding, not unseen-person inference without calibration.

Outputs: `protocol.json`, `frozen_selection.json`, `confirmation_metrics.json`,
`confirmation_config.json`, and per-person model/prediction artifacts under
`outputs/decoder_validation_001/`. The streaming adapter rejects a per-person
model paired with a different EEGBCI subject filename.
