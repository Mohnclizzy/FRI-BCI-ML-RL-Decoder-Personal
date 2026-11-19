# Phase 2 · Learning EEG request timing across people and days

The release at `019a7d1` is complete. The owner has asked to pursue a larger scope using additional time and GPU resources. This brief turns the larger-training option in the [roadmap](ANTIGRAVITY_ROADMAP.md) into the next development assignment. The proposed study and its gains remain unmeasured.

## Research aim

Test whether neural pretraining across people and days improves detection of motor-imagery requests after a fixed amount of calibration from a new person. Measure early episode detection, missed episodes, false starts during REST, and every transition request. The released classical controller is the reference. Successful request sequences can then drive the existing robomimic lift policy, with identical robot starts and acceptance rules.

The previous GPU campaign selected cue-window classifiers on PhysioNet. The later irregular-timing pilot fitted a classical decoder per person. Broad neural pretraining for timely requests across recording days remains an open comparison. [Training campaign](../results/gpu_campaign_2026-11-13/summary.md) · [Published pilot](../results/bciciv_ds1_request_pilot_2026-11-19/summary.md)

## Work package 1 · Make the larger corpus usable

Start downloading and inspecting representative Stieger recordings. The published corpus contains over 600 hours, 62 participants and 598 sessions. Inspect two sessions from one person and one session from another. Put these people in development before looking at their signals. Verify actual sample counts, event codes, montage, sample rate, breaks and available trial phases. Produce a manifest with original URLs, sizes, hashes and task meanings. Download the remaining eligible training data once the parser and storage estimate work. [Original dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8016873/)

The source task uses voluntary rest as a command to move a cursor down. Preserve commanded relaxation, intertrial rest and unlabelled time as distinct annotations. Establish whether the stored data support continuous replay or only trial-level decoding. Preserve original discontinuities. Measure the usable hours after processing. Account for visual cue and feedback signals when interpreting results.

Build an adapter that emits EEG, original timestamps, channel names, subject/session identifiers and scoring annotations. The model receives EEG available by its prediction time. Keep annotations out of prediction and controller state. Save raw data and training caches outside Git with a recovery manifest.

**Deliverable** — one runnable adapter, three audited recordings, a cohort inventory and a written choice of training labels and valid evaluation endpoints.

## Work package 2 · Freeze the comparison

Reserve people for source training, development and testing before model selection. Use a deterministic split recorded by participant ID. For test participants, designate the earliest eligible session as the fixed calibration allowance and later sessions as evaluation. All methods receive the same calibration examples. Fit preprocessing and early stopping from their assigned training/development data.

Compare three methods on identical recordings.

1. The classical Riemannian decoder with the released request controller.
2. One compact causal neural decoder trained from scratch on the calibration allowance.
3. The same neural architecture pretrained on source people, then adapted using that allowance.

Choose one architecture after a development smoke run. Reuse the current EEGNet implementation where suitable. Use genuine trailing EEG and measure whether a shorter observation window helps detection, with one candidate duration selected on development. The existing Berlin recordings can support diagnosis and development. Their already inspected outcomes are historical comparisons.

Use later Stieger sessions to test cross-day decoding for the labels they actually supply. For the stronger claim of improved asynchronous request timing, designate independent continuous recordings with irregular timing and labelled idle periods. Record both evaluation purposes explicitly if they require different datasets.

**Deliverable** — versioned split/config, metric definitions, fixed controller calibration procedure, a numeric minimum worthwhile gain and allowed false-start/transition regressions selected before test scoring.

## Work package 3 · Run a measured GPU campaign

Use one H100 for a short end-to-end run. Record data-loading throughput, GPU memory, training time, validation scores and estimated full-run time. Fix an input bottleneck before scaling hardware.

Pretrain on nested 10%, 30% and 100% subsets of eligible source participants, with three training seeds. Use a matched optimizer-update budget for the data-scale comparison and report actual examples and updates. A separate convergence run can answer whether extra optimization helps. Select the final recipe on development.

Use multiple GPUs for independent scales and seeds when this reduces completion time. Distributed training of one model is available if profiling justifies it. Store configs, checkpoints, source manifests and resume state with hashes. Launch paid jobs with a concrete instance, duration estimate and spend limit consistent with the owner's authorization.

**Deliverable** — a development learning curve showing what extra data bought, a reproducible selected checkpoint, and a locked evaluation command.

## Work package 4 · Evaluate once and decide

Run the locked comparison on reserved recordings. Report each person and session, calibration minutes, achieved REST false starts per minute, detection by elapsed time, total episode detection, delays and misses, and post-start/post-stop requests. Keep unknown transitions visible. Include time-only controls. Treat seeds as repeated training runs for the same people.

Advance a candidate to matched robot replay when it meets the predeclared criteria against the classical baseline. Report rejected requests, intended and unwanted initiations, and continuous retention. Use the original physical criteria and identify any change in scheduling as its own experiment.

**Deliverable** — a second study explaining whether additional training data improved useful command timing, where improvement held, and how much calibration it required. A failed comparison still closes this study with measured limitations.

## Session handoff

Antigravity should begin work package 1 now and continue through available CPU work without requiring a new prompt for each file. At each completed package, record outputs, verification, current revision and the next executable action at the top of STATUS. Commit and push code and compact evidence using the existing workflow. Keep the released report and evidence recoverable at their original commit. Record the GPU run configuration before provisioning.

The first session should leave a source manifest, inspected example recordings, a working parser and the next ingestion command. The first GPU session should leave a working training run and measured throughput. Avoid declaring either task complete based on a plan alone.
