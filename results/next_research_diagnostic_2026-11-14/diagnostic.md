# Saved-stream diagnostic and follow-up decision

## Scope

This diagnostic reuses the completed R08 command study's saved continuous probabilities and annotation intervals for S001–S015. It does not retrain, refit a gate, or load R12. The predeclared stream analysis classifies each prediction by whether its two-second trailing EEG window lies within a state or crosses a state boundary. Labels are used here only after scoring, for error analysis. EEGNet's three training-seed values are averaged within participant; cohort summaries are participant means and SDs (N=15). CSP and Riemannian have one fitted model per participant.

Reproduce with:

```powershell
python eeg_decoder\run_next_research_diagnostic.py
```

Machine-readable participant values are in `participant_diagnostic.csv`; group means are in `diagnostic.json`; the figure is `transition_scores.png`.

## Results

| Decoder | REST interior hand score | Imagery→REST transition | Imagery interior hand score | REST→imagery transition | Post-imagery REST episode maxima in first 2 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEGNet, three seeds averaged per person | 0.347 ± 0.136 | 0.479 ± 0.096 | 0.548 ± 0.137 | 0.406 ± 0.126 | 67.9% ± 16.0% |
| CSP/LDA | 0.317 ± 0.256 | 0.441 ± 0.188 | 0.533 ± 0.225 | 0.375 ± 0.254 | 65.2% ± 18.5% |
| Riemannian/LDA | 0.323 ± 0.272 | 0.465 ± 0.203 | 0.586 ± 0.250 | 0.371 ± 0.292 | 69.0% ± 16.1% |

Entries are mean participant score ± participant SD; peak percentage is calculated per participant before cohort aggregation. A score of 0.5 is the model's hand-probability midpoint. It is not a calibrated estimate of intended task probability.

The original study's 20-second selected-command stream had 38.9% of annotated REST time enabled; raw commands had 51.0%. The gate reduced false activation but also reduced imagery coverage and had three terminal-retention failures, all for participant S006 across three robot seeds. The transition decomposition supplies a plausible contributor: for all three decoders, roughly two-thirds of post-imagery REST episode score maxima occur in the first two seconds after REST begins, when a two-second trailing input still contains prior imagery. This is not the entire problem. Mean hand score in the pure imagery interior is only 0.55 for EEGNet, while REST interior is 0.35; their broad overlap and large person-to-person variability leave limited clean separation. The reverse transition also has a lower hand score (0.41) than pure imagery, consistent with expected trailing-window delay.

The fixed gates change both coverage and false activations, but no threshold fixes overlapping state evidence or creates missing imagery information. A persistence gate can further delay commands or carry them across an annotated boundary; the underlying imagery→REST scores already remain elevated. The right conclusion is a mixed bottleneck: trailing-window transitions account for many REST peaks, while weak and highly variable interior separation also limits performance. It would be inaccurate to describe this as a gate-only problem or to claim the dense-window training hypothesis is proven.

The perfect-command reference proposed in the handoff has not yet been run. It would be a useful low-cost simulator diagnosis, but it would remain privileged because it reads cued recording labels. A replay of periodic cue labels is not a self-paced intent experiment: it cannot establish that a participant independently chose to issue a request.

## Decision

1. Do not launch another broad GPU search. Existing campaign evidence already compares 54 recipe groups and 2,430 target checkpoints; the selected EEGNet still has poor command tradeoffs.
2. Do not claim that a repeated request wrapper evaluated on the same cue-paced PhysioNet R08 labels tests spontaneous task initiation. Existing R08 is already used for gate selection, and target R12 plus the source pool have been inspected/trained on. A wrapper can be smoke-tested on the simulator, but results on these recordings remain exploratory.
3. If time permits a bounded offline training follow-up, conduct only the matched dense causal-window ablation in the handoff. Train on training recordings alone, score complete held-out recordings, retain all transitions, and use the fixed request controller. Treat the result as exploratory; densely sampled overlapping windows are not independent evidence.
4. The most persuasive next research study requires a new prospective self-paced motor-imagery recording with an explicit “request lift” instruction, uninterrupted REST intervals, known request onset, enough idle time to measure false starts per minute, and held-out participants or sessions. Pair it with the same asynchronous controller, a small transparent baseline, and an autonomous robot task. Freeze the controller on separate development data and evaluate once on unused confirmation data. If prospective collection is infeasible, write up the current study and this limitation instead of representing cue replay as self-paced control.

## Research overlap and novelty check

The broad ideas of continuous motor-imagery detection, dwell/refractory filtering, task-level shared autonomy, false-positive reduction and asynchronous request initiation have prior primary literature. The present implementation does not justify novelty claims for any of those ideas by itself. See the standalone [novelty matrix](novelty_matrix.md).

| Primary source | Relevant method/result reported by source | Overlap with this project | Implication |
| --- | --- | --- | --- |
| Townsend et al. (2004), [Continuous EEG Classification During Motor Imagery—Simulation of an Asynchronous BCI](https://doi.org/10.1109/TNSRE.2004.827220) | Continuously classified motor imagery versus idling and introduced dwell time and a refractory period to trade hits against false detections. | Continuous score streams and temporal command filtering. | Asynchronous MI detection plus dwell/refractory rules are established ideas. |
| Song et al. (2022), [An EEG-based asynchronous MI-BCI system to reduce false positives with a small number of channels for neurorehabilitation: a pilot study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9510756/) | Evaluated asynchronous MI false-positive rejection online with healthy participants and stroke patients; reported a conservative system with a sensitivity tradeoff. | Explicit false starts/false activations in asynchronous MI. | False-positive reduction is already a direct research objective in the field. |
| Liu et al. (2019), [Shared control of a robotic arm using non-invasive brain-computer interface and computer vision guidance](https://doi.org/10.1016/j.robot.2019.02.014) | Combined MI-BCI with computer-vision-guided autonomy so binary BCI output could direct an arm task. | MI-triggered robot action with autonomous low-level control. | “EEG starts an autonomous robot skill” is not novel as a broad system architecture. |
| Akinola et al. (2017), [Task Level Hierarchical System for BCI-Enabled Shared Autonomy](https://doi.org/10.1109/HUMANOIDS.2017.8246878) | Used SSVEP for high-level task requests while a robot handled navigation and manipulation. | Task-level request and shared autonomy. | Relevant architecture precedent; different EEG paradigm and task. |
| Wu et al. (2024), [Motor Imagery Classification for Asynchronous EEG-Based Brain-Computer Interfaces](https://doi.org/10.1109/TNSRE.2024.3356916) | Proposed a sliding-window MI prescreen plus task classifier and evaluated within- and cross-subject results on four EEG datasets. | Continuous sliding-window deployment and mixed state detection. | Causal sliding-window classification is active prior work; a dense-window training change alone is unlikely to be novel. |
| Dillen et al. (2024), [User Evaluation of a Shared Robot Control System Combining BCI and Eye Tracking in a Portable Augmented Reality User Interface](https://pmc.ncbi.nlm.nih.gov/articles/PMC11359122/) | User study of shared robot control combining BCI and eye tracking, with repeated classifier evidence used to authorize actions. | Request confirmation and autonomous shared robot control. | Evidence accumulation plus robot initiation has close conceptual overlap. |

This is a bounded positioning review, not a systematic literature search. It is enough to reject novelty claims for the broad architecture or a basic sequential gate. Before submission, a researcher should verify full texts, identify the precise incremental contribution, and broaden the review to current primary work. The more defensible contribution available is a carefully measured analysis of how cue-window-trained decoders fail during causal state transitions and what that implies for a specific task-level controller. The current reused R08 recordings support an exploratory methods report, not an independent confirmation claim.
