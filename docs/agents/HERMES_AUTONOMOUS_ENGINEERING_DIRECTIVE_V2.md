# Hermes Autonomous Engineering Directive v2 — reconciled 2026-11-04

This revision reconciles the Brev Codex CLI history, published results, and newer Hermes artifacts. It supersedes the earlier v2 startup/status text; the filename remains stable for existing references.

## Mission and ownership

Hermes, using local Qwen/Ollama on Brev, is the active engineering/execution owner. Codex/ChatGPT reviews evidence, code and experimental design. The project owner sets priorities. Do not start competing experiment writers or duplicate jobs. A reviewer reading saved files does not transfer execution ownership.

Deliver the already-authorized simulation demonstration: recorded motor-imagery EEG enables, holds and resumes a learned Franka table-pickup skill, followed by an honest matched gate comparison. Robot learning remains required. The official pretrained robomimic BC-RNN route is already authorized and implemented; do not restart Isaac PPO solely because an older directive names Isaac. Preserve and attribute the existing Isaac PPO results and the external pretrained checkpoint separately.

## Canonical repository and startup order

Source: Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal.
Canonical working/results branch: codex/robot-learning-design.
Brev execution checkout: /home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal.

The copy of this directive on codex/isaaclab-project-scaffold is a discoverability mirror, not an instruction to execute from that branch. For the latest canonical copy use:
https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal/blob/codex/robot-learning-design/docs/agents/HERMES_AUTONOMOUS_ENGINEERING_DIRECTIVE_V2.md

At startup or recovery, read:
1. AGENTS.md and docs/agents/CODEX_HERMES_RECONCILIATION_2026-11-04.md.
2. Latest STATUS.md, checking its observation time against actual processes and newer artifacts.
3. docs/PROJECT_EXECUTION_HANDOFF_2026-11-03.md and docs/SESSION_WRAP_2026-11-04.md. The wrap describes the earlier Codex stop, not current Hermes inactivity.
4. Latest completed and active phase protocols, source, metrics and raw-trace locations. Start with results/robomimic_latch_2026-11-04/ and the newer local results/robomimic_latch6_zoffset_2026-11-04/.
5. README.md, docs/research_protocol.md, and docs/literature_reference_summary.md before changing experiments. Treat older README/startup milestones as historical where newer evidence supersedes them.
6. Git branch/HEAD/remotes/status, live processes, GPU usage and saved run registry.

Resolve missing local documents from the explicit design branch through available authenticated tools into a separate reference location. The dirty Brev checkout was still at synthetic/local HEAD b3422a0 during reconciliation, while published results were newer. Do not pull blindly, reset, clean, force-push, or change its branch. Use an isolated checkout for Git integration and preserve executed source snapshots.

## Corrected evidence baseline

- The corrected 120-second scripted EEG comparison is COMPLETE for S006-S010/R12: raw 5/5 lifts, confidence 3/5, temporal 2/5. See results/corrected_eeg_2026-11-03-complete/summary.md. The older interrupted batch and invalid 24-second comparison are separate historical records. This is scripted control, not learned EEG control.
- Isaac unilateral PPO: 3/5 retained near-table diagnostic starts. Half-arm inference ablation: 0/5. Do not repeat rejected routes without new evidence.
- Native pretrained BC-RNN Phase 2: lift event 5/5, terminal retention 0/5.
- Phase 3 acquisition-pose HOLD diagnostic: 5/5 retained, with narrow height margin; not continuous-control qualification.
- Phase 4: HOLD target/timing checks passed, but physical retention/resume failed.
- Phase 5: 24 matched runs; 21/21 positive acquisitions, 17/21 retained. Four lost the required height while contact continued. The physical gate failed; learned EEG replay did not run.
- Hermes Phase 6: newer local +0.015 m completion-goal intervention produced 24 saved runs. The reconciliation audit found 21/21 positive retention and 0/3 negative acquisition, with the listed trace consistency checks passing. This is a development result on reused seeds, not held-out qualification or a completed learned EEG demonstration. See the reconciliation report for audit scope and unresolved protocol/publication/backup checks.

A proposed run, background-process completion notification, reward or exit code is not a verified result. Preserve contradictory/stale records with dated corrections; do not erase earlier failures.

## Current continuation boundary

Do not rerun the Phase 6 batch merely because STATUS still says Phase 5. First inspect the existing Phase 6 source and 24 traces, finish the full audit against the frozen Phase 5 criteria, and recover evidence of the pre-run protocol. At reconciliation, protocol6.md was referenced by the analyzer but absent in the inspected phase directories. If no pre-run protocol exists, disclose that gap and document the reconstruction as retrospective; never backdate it.

Preserve exact source, checkpoint, commands, hashes, all attempts and failures. Publish compact Phase 6 results and update STATUS; back up raw traces/video through the existing private artifact workflow and verify downloaded hashes. Check current publication authentication without exposing credentials.

Once the authorized physical-control gate is established, the existing next task is one exploratory 120-second S006R12 raw/confidence/temporal learned-policy replay with always-enable/HOLD controls, a declared label-independent reset schedule, causal timestamps and fixed budgets. Keep later held-out qualification separate. Reconcile any newer completed work before starting this step.

## Controller and scientific invariants

Before completion, REST or stale input retains HOLD priority. HOLD keeps appropriate absolute equilibrium goals and gripper state fixed, makes no policy calls, and preserves explicit recurrent-state behavior while physics, EEG clocks and deadlines continue. Resume uses current observations. Completion-latch behavior must remain explicit in reports; carrying requests after completion do not test active learned-policy resumption.

The +0.015 m Phase 6 completion target is a one-time declared post-acquisition intervention, not permission to drift or rewrite ordinary HOLD goals. Maintain the existing 8 cm rise/contact/retention rule and 400-step native-control horizon for matched controls. Contact sampled at 20 Hz cannot establish absence of every physics-substep break. Keep the historical runtime compatibility limitation visible.

Reuse the existing causal decoder/gate contracts. No annotation labels or future samples enter the decoder, gate or actor. Separate EEG time, physics time and enabled policy steps. Report completion, rest motion/activation, misses, delay, switches and drops together. Do not tune until a preferred filter wins. Offline fist-imagery labels do not establish grasp intent, live usability or hardware safety.

## Work loop and authority

Recover evidence -> identify one next measurable question -> carry out already-authorized bounded work -> inspect artifacts -> preserve/publish -> update status -> continue.

The previous read-only startup audit has already occurred. Do not reset the project to a fresh audit-and-approval loop solely because this file was reread. Continue within the owner's existing task/resource permissions. Routine fixes, tests, documentation, feature-branch commits and authorized bounded native/Isaac experiments do not require repeated approval. This reconciliation does not authorize larger resource budgets, destructive actions, credential changes, public sharing, protected-branch merges, or new scientific assumptions.

Never modify Angel's repository. Never overwrite evidence, publish secrets or copy account credentials between agents. Preserve /home/ubuntu/FRI-project-8b9db42 and all previous checkpoints/run directories.

Use STATUS.md as the concise current phase/job/result/blocker/next-action record. Use existing per-phase protocols/metrics plus a compact run registry; avoid competing status documents. If docs/agents/AGENT_STATUS.md, ENGINEERING_BACKLOG.md, EXPERIMENT_REGISTRY.md or HANDOFF.md exist, make them consistent pointers or clearly scoped records. GitHub holds durable code/compact evidence; private storage holds large artifacts; local transcripts support recovery but are not shared conversational memory.

After interruption, inspect processes and saved outputs before restarting. Report exact blockers, preserve portable results, and continue independent authorized work. Do not leave jobs untracked or claim completion because a budget ended.
