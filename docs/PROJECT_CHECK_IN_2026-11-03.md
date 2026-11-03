# Project check-in — 2026-11-03

## Current state

The project has separate research components for public EEG decoding, causal
command replay, scripted Franka behavior, and learned policy development. It
does not yet have a qualified learned table-pickup controller connected to EEG.
There is no live EEG, hardware, or clinical result.

## Robot learning

The initial pre-grasp checkpoint lifted from in-hand starts in its five-episode
screen; the transferred table checkpoint did not pick up the table cube. The
later near-table progress variant was trained from scratch for 250 iterations
with 256 environments (2,048,000 transitions). Its five 12-second physical
screens for seeds 8–12 met the existing physical lift definition in 2/5 episodes
(seeds 10 and 11); seeds 8, 9 and 12 did not. Seed 10 recorded one drop. This is
a fixed-scene development diagnostic; repeated seed values do not establish
spatial generalization, and the proposed 18/20 readiness gate is unmet.

A separate fixed-open playback task retained the policy's nine-action interface
and sent both finger targets to 0.04 m while passing the seven arm actions
through unchanged. The one-step GPU action check passed. Two fresh-process,
one-environment, 30-policy-step runs (seeds 8 and 11) each had an explicit
incomplete end state, full 120/120 contact coverage and no physics gaps. Actual
finger targets were 0.04 m on every physics sample. Both produced right-finger
contact >=1 N (0.0667 s and 0.1000 s) with peak right force 92.35 N and 169.28 N;
left-finger contact remained 0 N. Maximum cube-center displacement was 0.8399 m
and 0.5533 m, while maximum cube rise was only 0.0165 m and 0.0027 m. These
selected one-second contact-mechanism diagnostics are not pickup-success
estimates. See [the fixed-open record](FIXED_OPEN_PLAYBACK_2026-11-03.md).

## Frozen scripted comparison

The scripted skill remains the reference: it has lifted a free table-supported
cube, and the corrected always-HOLD, uninterrupted, and pause/resume controls
passed their frozen checks. The historical 24-second five-person physical
comparison is invalid because its always-HOLD control drifted. An earlier
corrected 120-second attempt stopped incomplete; a later batch completed all
15 gate/person episodes and the three frozen controls. Its exploratory paired
outcomes are reported below. The scripted comparison remains separate from
robot learning.

## EEG and HOLD interface

The frozen per-person CSP/LDA selection scored 59.33% balanced accuracy over
150 held-out-run windows for S006–S010; rest-window false activation was 29.33%.
The corrected 120-second physical scripted comparison is now complete for all
five run-12 recordings and all three controls passed. Raw gates produced lifts
in 5/5 cases, confidence gates in 3/5, and temporal gates in 2/5. Mean executed
command activation during annotated rest fell from 38.7% (raw) to 25.7%
(confidence) and 19.6% (temporal); measured physical motion during rest fell
from 13.5% to 13.2% and 9.1%. Missed imagery bouts increased from 12 (raw) to
30 and 38. This exposes the practical tradeoff in this exploratory replay:
filtering reduced unwanted activation and motion, while reducing task
completion and imagery response. These measures come from a fixed simulated
scene and offline cue labels; they do not establish live EEG or hardware
performance. See [the paired results and figure](../results/corrected_eeg_2026-11-03-complete/summary.md).

The intended learned-policy interface remains continuous enable/HOLD/resume:
REST or stale input retains HOLD priority, and the robot policy owns low-level
actions. A learned-policy HOLD/resume adapter has not been implemented or
validated. No recorded EEG has controlled a qualified learned policy. The data
labels remain instructed rest versus left/right fist imagery combined as hand
imagery; they do not identify grasp intent.

## Next bounded work

For robot learning, the fixed-open result rules out finger closing commands
as the sole cause of the early right-only contact. Inspect the actual fingertip
collision poses and arm path against the scripted reference before allocating
another PPO run. The two one-second traces do not estimate pickup reliability.

Next, use this tradeoff to choose a single EEG gate for a clearly labeled
follow-up, with imagery misses and rest motion both as acceptance measures.
For robot learning, inspect the arm path and contact geometry before another
PPO budget. Keep these results separate until the learned HOLD/resume adapter
and pickup controller have their own physical evaluation.

## Source and execution note

The Brev CLI worked in a separate worktree based on its locally available
`b3422a0` checkout. It could not fetch the newer Personal design branch because
its GitHub credentials were unavailable, so it left commit `a1c056c` locally.
The code and compact results were subsequently reviewed and integrated into the
current Personal design branch from the Windows workspace. The executed task,
near-table configuration and recorder source match the integrated content after
newline normalization; the task registration in the current branch also retains
the other existing tasks. Raw traces and the action check were copied off Brev
and verified by SHA256. The original dirty Brev checkout was preserved.
