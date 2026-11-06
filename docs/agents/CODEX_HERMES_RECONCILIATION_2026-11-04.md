# Codex / Hermes reconciliation — 2026-11-04

## Finding

The disconnect was stale and split project memory, not missing CLI history. This chat recovered Brev's saved Codex sessions directly over the existing SSH connection, checked current repository/run files, and reconciled the Hermes directive. No model login was changed and no robot rollout or training job was launched by this review.

The installed CLI is codex-cli 0.166.0. Its login-status command reports "Logged in using ChatGPT"; that does not establish that its account is identical to this desktop account. Credentials were not copied or displayed. Reviewing saved history did not require switching accounts.

## Recovered history

Saved transcripts remain in /home/ubuntu/.codex/sessions/ and /home/ubuntu/.codex-cli-runtime/sessions/. The accompanying evidence.json inventories and hashes 13 Nov 3-4 session files without exporting their raw messages.

| Session | Role and verified endpoint |
|---|---|
| 01a1a399-04ff-7590-b586-cc55c86386df | Nov 3 Isaac work: unilateral 3/5; half-arm 0/5 rejected |
| 01a1a49a-1df1-7b00-83c8-de4a2050f9aa | Main Nov 4 execution owner; user requested wrap-up at 05:19:25 UTC |
| 01a1a4b7-2583-7bd3-ac12-513f96e17728 | Phase 1 relevant off-host preservation |
| 01a1a4c1-1e9c-7d93-9c53-86440048267b | Phase 2 native BC-RNN: 5/5 lift events, 0/5 terminal retention |
| 01a1a4cf-02ab-7ab3-b99f-90fc0234b887 | Phase 3 post-acquisition HOLD: 5/5 diagnostic retention |
| 01a1a4df-3a88-7691-aa2f-3caadb5d90ca | Phase 4 continuous adapter: invariants passed, physical gate failed |
| 01a1a4f4-0bc3-7073-9651-b703de4e08a3 | Phase 5 latch: 21/21 acquired, 17/21 retained |
| 01a1a4ff-2781-7d70-aa8b-ec220b3cd027 | Phase 6 instruction and inspection only; explicitly interrupted at 05:19:39 UTC |

Correction to the prior conversational summary and published wrap wording: the exact CLI event is a user-requested interruption during wrap-up, not evidence that a usage-limit failure terminated this Phase 6 turn. The local wrap agrees. No completed Phase 6 experiment appears in that Codex session. Hermes's later Phase 6 files must be attributed separately.

## Newer Hermes evidence

At the 10:21 UTC inspection, /home/ubuntu/robomimic-native-phase6-20261104/ contained two smoke cases and all 24 matched controls. Source was under results/robomimic_latch6_zoffset_2026-11-04/. Hermes/Ollama processes were present; no matching robot rollout process was observed in the process snapshot. These are time-stamped observations, not a permanent "no job" claim.

The saved-artifact audit found:
- 21/21 positive controls acquired and retained without a sampled post-latch contact/height loss; all three always-HOLD controls remained negative.
- Positive terminal rise ranged approximately 0.09222-0.10259 m.
- Every trace had 401 sequential states through step 400 at the recorded 20 Hz clock.
- Sampled bilateral contact agreed with the simulator's built-in grasp signal.
- Locked intervals had no policy-call or recurrent-counter advance; latch goals/orientation and gripper commands matched their saved fixed values.
- Initial cube and end-effector positions matched corresponding Phase 5 traces. This does not establish identical complete simulator state.
- Saved summary HOLD/latch invariant flags passed. The independent audit did not recompute every HOLD drift metric.

Run the companion audit_saved_state.py with a new output JSON path on Brev to reproduce this consistency check. The snapshot evidence.json contains per-trace SHA256 values and source hashes. This is review of existing data, not a new experiment or full qualification sign-off.

Outstanding: no protocol file was found in the inspected Phase 6 source/raw directories, although analyze6_batch.py references protocol6.md. Recover any pre-run declaration from the agent's saved records; otherwise disclose the gap and label later documentation retrospective. Full Phase 6 publication, video and off-host recovery were not verified by this review. Contact is sampled at 20 Hz; substep breaks remain unobserved. Seeds 0-2 are reused development cases. Historical mujoco-py compatibility remains a limitation.

## Directive corrections

1. Canonical results/implementation branch is codex/robot-learning-design. The old directive existed only on codex/isaaclab-project-scaffold; its discoverable copy now identifies the canonical branch.
2. The corrected scripted 120-second S006-S010/R12 batch is complete: raw 5/5, confidence 3/5, temporal 2/5 physical lifts. See results/corrected_eeg_2026-11-03-complete/summary.md. It remains separate from learned-policy EEG replay.
3. Native robomimic is an already-authorized route, not an unapproved deviation from an Isaac-only mission.
4. Read current status, latest artifacts and the execution handoff before Nov 2 historical files. A stale "paused/no job" paragraph must not trigger rerunning completed Hermes work.
5. Hermes remains execution owner. Codex is reviewing; no second experiment owner was started.
6. Existing bounded-work authorization remains in force; reading the directive again does not reset the completed startup audit.
7. Missing local documents must be retrieved from the explicit branch without resetting the dirty canonical checkout.

The Brev checkout remained dirty at b3422a0 with newer untracked experiment files. GitHub CLI authentication check exited 1 during this review; therefore earlier plans to authenticate gh are not proof publication works from Hermes. The connected GitHub tool remains available from this chat. No credentials or Git settings were changed.

## Exact next handoff

Hermes should finish auditing/documenting/preserving the existing Phase 6 batch before deciding whether its physical gate permits the already-planned exploratory S006R12 EEG replay. Do not rerun the batch merely to synchronize status. Preserve Phase 5 failures and the distinction between learned acquisition, controller completion latch, scripted EEG reference, exploratory replay and held-out qualification.

Primary records:
- [Current status](../../STATUS.md)
- [Execution plan](../PROJECT_EXECUTION_HANDOFF_2026-11-03.md)
- [Prior session wrap](../SESSION_WRAP_2026-11-04.md)
- [Reconciled directive](HERMES_AUTONOMOUS_ENGINEERING_DIRECTIVE_V2.md)
- [Phase 5 result](../../results/robomimic_latch_2026-11-04/summary.md)
- [Corrected scripted comparison](../../results/corrected_eeg_2026-11-03-complete/summary.md)
- [Audit evidence](reconciliation_2026-11-04/evidence.json)
- [Audit script](reconciliation_2026-11-04/audit_saved_state.py)

The original transcripts, historical reports, experiment source and outputs were preserved. This reconciliation adds a dated correction instead of silently rewriting earlier evidence.
