# Project history and current research record

This file is the durable, cross-chat summary of the decisions, iterations,
measurements, and unresolved work in this repository. Git commit history records
the implementation steps; this document records why the direction changed and
what the evidence currently supports. It is a curated project record, not a
verbatim export of ChatGPT conversations.

## Project in one paragraph

The project studies a simulation-first EEG-to-robot control pipeline. Public
PhysioNet EEG Motor Movement/Imagery data is decoded into REST or hand-imagery
requests, mapped to HOLD or GRASP_LIFT, and replayed against a simulated Franka
cube-lifting task. The central comparison is raw requests versus confidence and
temporal filtering, measured by physical task completion, unwanted movement
during rest, missed commands, and delay. EEG labels describe instructed fist
imagery; they do not identify a particular grasp. A learned robot policy and an
EEG decoder are separate components, and EEG has not yet been shown to control
a learned robot policy successfully.

## Major iterations

### 2026-10-12 — EEG decoder and research foundation

- Initialized the project with a public EEG motor-imagery dataset, a pooled
  CSP/LDA baseline, reproducible data preparation, and research documentation.
- Kept rest and fist imagery as the valid labels. Avoided claiming that this
  dataset labels grasp, wrist flexion, or another specific hand action.
- Defined recording-level held-out evaluation and recorded the literature
  context. See `docs/research_protocol.md` and
  `docs/literature_reference_summary.md`.

### 2026-11-01 — Decoder comparison and causal recorded-EEG replay

- Added calibrated per-person decoder comparisons and froze a fresh-cohort
  confirmation protocol before inspecting its results.
- Recorded the five-person confirmation result: per-person CSP reached 59.33%
  balanced accuracy after calibration. This cohort is now inspected, so later
  experiments on it are exploratory rather than fresh confirmation.
- Added offline replay that preserves causal filtering state across chunks,
  carries timestamps and source annotations, and evaluates REST/imagery command
  behavior on held-out recordings.
- Established the command vocabulary and separated decoder predictions from
  accepted robot actions. The command metrics are not physical robot outcomes.
- Added an Isaac Lab simulation scaffold and bounded cloud workflow; established
  that simulator setup and decoder evidence need separate validation.

### 2026-11-01 to 2026-11-02 — Scripted physical lift and RL exploration

- Built a fixed-scene scripted Franka controller and verified that it lifted a
  free cube about 0.179 m with sustained bilateral contact. This is a scripted
  control demonstration, not learned-policy success.
- Explored PPO first from a privileged pre-grasp start, then a table pickup.
  Pre-grasp training success was weak and table transfer did not establish a
  successful lift. Training reward and pre-grasp starts are not evidence of
  table-pickup success.
- Audited PPO observations against simulator state and recorded reward terms to
  guide a bounded reward follow-up. A reward change remains an experiment, not
  an established improvement.

### 2026-11-02 — Physical controls, executor audit, and interrupted comparison

- Audited the first matched EEG physical comparison and found that HOLD had
  followed changing measured positions, allowing arm drift. The earlier 1/5 raw
  lift and 0/5 filtered lifts are retained as debugging history, but are not
  valid evidence that filtering changed physical completion.
- Repaired HOLD target behavior and added a separately measured initialization
  interval. After repair, the always-HOLD negative control stayed within the
  frozen displacement threshold. Uninterrupted and paused/resumed scripted
  table lifts passed as positive controls in the fixed scene.
- Ran a 120-second temporal-filtered S006R12 integration case. Request/trace
  alignment and telemetry coverage were validated, but the gate admitted only
  0.692 active seconds, the robot did not move or lift, and annotation scoring
  detected 1 of 15 imagery bouts. This is one integration case, not a matched
  filter comparison or evidence of a filter benefit.
- Started the corrected five-person, three-gate matched batch and stopped it
  before completion. Its partial output is explicitly marked incomplete. The
  final paired comparison, plot, report, and short demonstration remain undone.
- No new PPO training began after the audit. The proposed 64-environment,
  100-iteration approach-reward follow-up is still unrun.

## Current evidence and boundaries

Use these documents as the authoritative short-form status, protocol, and
results index:

- `START_HERE.md` — plain-language orientation and next task.
- `docs/stopping_point_2026-11-02.md` — exact last measured state, evidence
  folders, and outstanding actions.
- `docs/completion_protocol_2026-11-02.md` — frozen corrected comparison
  protocol and acceptance criteria.
- `docs/multi_person_replay_2026-11-02.md` — command-only five-person replay
  results and limitations.
- `docs/end_to_end_pilot_2026-11-02.md` — integration pilot details.
- `docs/ppo_lift_evaluation_2026-11-02.md` and
  `simulation/cloud/pregrasp_pilot_2026-11-02.md` — policy training and
  physical playback evidence.
- `simulation/cloud/controlled_lift_2026-11-02.md` — scripted lift and
  physical-control details.

Do not claim that temporal filtering improves physical completion, that PPO
successfully lifts from a table, or that EEG controls a learned policy. The
corrected matched batch is the immediate unresolved experiment. Consult the
stopping-point and completion-protocol documents before resuming it.

## Where the history and files live

- Git commits preserve dated implementation milestones and the source/doc
  changes on branch `codex/isaaclab-project-scaffold` in Mario's personal fork.
- `origin` points to the independent personal repository
  `Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`; `upstream` points to Angel's
  original `AL-DEV06/FRI-BCI-ML-RL-Decoder`. A GitHub fork was also created
  during setup, but it remains tied to the original private repository's
  permissions. The independent repository is the one to share and update.
  Push ongoing work to `origin`; fetch `upstream` only when intentionally
  checking for updates from the original repository.
- Current source and handoff documents are tracked in this repository.
- `data/` and `outputs/` are intentionally ignored by Git. The public raw EEG
  dataset can be downloaded again using the documented commands. Local run
  artifacts, including checkpoints and detailed telemetry, stay in the local
  `outputs/` folder. Because this checkout is under OneDrive, those files can
  be shared through OneDrive folder permissions; they are not included when
  someone clones this Git repository.
- `.venv/` and caches are local machine state and are not shared. Recreate the
  environment from the requirements files.
- The separate `eeg_dataset_research.md` workspace note has been copied into
  this repository as `docs/eeg_dataset_research.md` for completeness.

For repository access and the second-account workflow, see
`docs/SHARING_AND_REPRODUCTION.md`.
