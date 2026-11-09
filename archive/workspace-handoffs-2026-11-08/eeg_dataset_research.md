# EEG dataset shortlist for the Robot Learning project

Research date: 2026-11-01. Read-only web review of original dataset records, papers, and official loaders. No raw recordings downloaded or tested; access below means a public record and file listing were found, not that file integrity or signal quality was established.

## Verdict

**For the movement-to-robot direction, inspect WAY-EEG-GAL first.** It combines real grasp/lift actions with EEG and physical measurements. It supplies the missing movement evidence in the current PhysioNet subset. This changes the main experiment from imagined commands to executed human movement; it cannot establish imagery-only control.

**No fully matched independent validation dataset is established for its entire trajectory/force target.** Ofner2017 offers related hand movements, but its motion channels have serious availability caveats. IACKD is a candidate for independent hand-trajectory testing after access, alignment, and task semantics are audited. Neither is plug-and-play validation of all WAY targets.

If preserving motor imagery is essential, use **BNCI2014_001 first and Cho2017 for independent left/right-imagery validation**. That pair supports decoder/calibration studies, not continuous human-motion reconstruction. Retraining the same method separately demonstrates repeatability; transferring a frozen model requires a separate protocol.

Better-matched measurements can expand the research question. They do not guarantee higher accuracy, and the earlier pooled PhysioNet failure is not proof that PhysioNet is bad data.

## Comparison

| Dataset | What the participant does | Measurements beyond EEG | Best use / important catch |
|---|---|---|---|
| WAY-EEG-GAL | Executes reach, pinch, lift, hold, replace, release | Hand/object kinematics, grip/load forces, EMG and events | First paired manipulation pilot; some series omit non-EEG measurements |
| Ofner2017 / BNCI001-2017 | Executes and imagines six right-arm/hand movements plus rest | Glove and exoskeleton sensor channels in the schema | Rich labels; most Armeo channels are empty in released runs |
| IACKD / ds006840 | Unimanual control under congruent/incongruent intention-action conditions | 3-D hand motion, stimulus stream, alignment derivatives advertised | Promising paired-motion candidate; archive content and licensing inconsistencies need inspection |
| BNCI2014_001 / BCI IV-2a | Imagines left hand, right hand, feet, tongue | EOG, cues, artifact information | Clear two-session evaluation; no measured imagined trajectories |
| Cho2017 | Imagines left/right hand movements | Four EMG channels and participant metadata | Many people for independent MI evaluation; one session |
| BNCI001-2020 reach/grasp | Executes palmar/lateral grasps with different electrode systems | Movement/grasp onset and offset events | Sensor-domain stress test; continuous kinematics not demonstrated in documented file variables |
| Orthosis error dataset, Zenodo8345429 | Feels deliberate wrong-direction motion while assisted | EMG and robot-error event markers | Optional feedback-learning branch; not motor-imagery data |

## Dataset audit notes

### 1. WAY-EEG-GAL — recommended first paired-motion inspection

Twelve people, 3,936 trials by collection description; 328 trials per person. Full data include measured motion, kinetics, EMG and timing. Different weights and friction conditions enable physical robustness questions. [Original collection](https://doi.org/10.6084/m9.figshare.c.988376)

EEG has 32 channels at 500 Hz; EMG is 4 kHz and other measurements 500 Hz. Participant ZIPs contain MATLAB structures, with author MATLAB/Octave utilities; about 15 GB of structures overall. Data/scripts are CC BY4.0. One weight series per participant withholds non-EEG information, so not every listed trial is usable for paired training. Verify separate device time axes instead of assuming matching rates imply alignment. [Data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/), [author utilities](https://github.com/luciw/way-eeg-gal-utilities)

Risks: movement/muscle and ocular artifacts, sensory feedback after contact, highly repetitive trajectories, and cue timing shortcuts. A time-since-cue predictor must be a baseline. No arbitrary finger articulation, imagined trajectories, or validation on novel robot actions is implied. A published benchmark may already have trained on these subjects; check pretrained-model overlap.

### 2. Ofner2017 — strongest fine movement labels, incomplete motion coverage

Fifteen people; two different-day sessions: execution first, imagery second. Six classes are elbow flexion/extension, forearm supination/pronation, hand open/close, plus rest. Ten runs/session and 60 trials/class/session. GDF contains 61 EEG channels at512 Hz, three EOG and sensor channels. Event codes and glove/exoskeleton schema are specified. [Original technical description](https://lampx.tugraz.at/~bci/database/001-2017/dataset_description.pdf)

Public participant ME/MI ZIPs, CC BY4.0, are listed by [BNCI](https://bnci-horizon-2020.eu/database/data-sets). The [official MOABB loader](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Ofner2017.html) documents that 270/300 runs have empty Armeo channels returned as zero; 20 contain arm kinematics, and subject1 MI has random placeholder channels replaced with zeros. Do not interpret zero-filled channels as real movement. Glove coverage and calibration still need raw-file inspection. Total download size unverified.

ME-versus-MI is confounded with recording day/order; this is not two repeated MI sessions. Great for assessing named movements, conditional generation and execution-to-imagery transfer with caveats; poor basis for promising a complete all-subject motion corpus.

### 3. IACKD — promising, provisional

The [OpenNeuro source mirror](https://github.com/OpenNeuroDatasets/ds006840) provides BIDS BrainVision files and events. Its dataset_description declares CC0. [NEMAR](https://nemar.org/dataset/on006840) reports15 people and roughly6.4GB archive/7.42GB dataset; catalog estimates vary, so reserve capacity and verify the pinned release.

The [authors' repository](https://github.com/Boketto1/IACKD) advertises7,040 trials (6,720 valid), EEG1024 Hz, Leap motion170 Hz, pre-movement segments and affine timestamp alignment with coverage/NaN tracking. Its README instead says data CC BY4.0 and still contains placeholder download links. Raw acquisition is described as CURRY while OpenNeuro distributes BrainVision. EEG channel counts differ across catalog records; verify per-recording sidecars. Exact session organization and complete kinematic archive availability remain unverified. Use the [data paper](https://doi.org/10.1038/s41597-026-07146-x) to settle fields before modeling.

This is conflict/visuomotor control, not natural grasp-and-lift or motor imagery. An affine synchronization fit is not automatically negligible timing error. No dedicated verified end-to-end loader for the full paired data was established; MNE-BIDS plus authors' alignment scripts are candidates.

### 4. BNCI2014_001 — easiest structured imagery baseline

Nine people, two sessions on different days,22 EEG+3 EOG channels,250 Hz, four imagery classes. Each session has six runs and288 trials. Cues and artifact flags exist; no actual hand motion is measured for imagery. [MOABB documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html)

[BNCI source](https://bnci-horizon-2020.eu/database/data-sets) lists public MAT downloads and **CC BY-ND4.0**. The source download timed out through the web tool, so file availability was not byte-verified. MOABB supplies a loader; original competition data are also [documented here](https://www.bbci.de/competition/iv/desc_2a.pdf). Total size unverified. Review ND terms before redistributing processed or synthetic derivatives; open access does not mean unrestricted redistribution.

Good for left/right matched evaluation, calibration curves and session shift. Familiar benchmark with possible pretrained-data overlap. A random split across correlated windows would undermine evaluation.

### 5. Cho2017 — many people, matching left/right labels

Fifty-two people, one session;64 EEG+4 EMG channels,512 Hz. Left/right hand imagery is accompanied by EMG to monitor unwanted physical movements. Five/six recording runs are described, although a loader may package them differently; inspect metadata rather than assuming loader run count equals acquisition run count. [Official loader documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html)

Data are MATLAB, hosted by [GigaDB100295](https://gigadb.org/dataset/100295), with an official MOABB downloader. [Original paper](https://doi.org/10.1093/gigascience/gix034). MOABB reports CC BY4.0, but the source record did not render readable terms in this review: **source license not independently verified**. Total size unverified. Strong independent left/right-imagery comparison with BNCI2a after montage/reference/rate harmonization; no repeated-day test and no measured movement trajectories. Separate subjects before preprocessing/model selection; do not feed EMG to a claimed EEG-only model.

### 6. BNCI001-2020 reach/grasp — electrode-type robustness option

Executed self-initiated palmar/lateral grasp:45 participants in three15-person groups using gel, water or dry electrodes. The original PDF reports58,32 or11 EEG electrodes respectively, with EOG on the first two systems. Eighty trials/condition over four movement runs; movement/grasp timing events are documented. MAT variables include signals, events and header; the documented schema does not establish continuous trajectories. [Original description](https://bnci-horizon-2020.eu/database/data-sets/001-2020/Dataset_description.pdf)

The BNCI listing has public participant files and CC BY4.0. Size, sample rates by device, repeated-day sessions and a dedicated Python loader were not independently verified. Different electrode systems use different people: hardware effects and participant effects are entangled. This is a sensor/domain stress test, not within-person proof that cheap hardware preserves performance. Exclude eye signals and pre-movement gaze shortcuts from EEG-only claims.

### 7. Orthosis error EEG/EMG — separate feedback branch

Eight people undergo assisted elbow flexion/extension with occasional250-ms direction reversals. Source archive version3 offers BrainVision EEG, EMG and metadata: EEG2.3GB, EMG358.9MB, metadata8.3kB; total2.7GB. MNE can read EEG. [Zenodo record](https://zenodo.org/records/8345429)

The [original paper](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1304311/full) specifies64 EEG channels500 Hz,8 EMG channels1000 Hz, synchronized triggers, and10 main runs/person. People squeeze a ball after detecting errors, introducing an explicit motor-response confound; rare-event/P300 effects can also matter. These are experimental phases rather than established repeated-day sessions. Raw robot trajectories were not verified. Dataset license was not exposed by the readable record; paper is CC BY, which alone does not settle the archive license. Do not confuse this dataset with a different movement dataset also called Kueper2024 in some catalogs.

## Research question and Mario's part

Candidate: **Does EEG add predictive information beyond cue timing and previous motion for controlling a grasp/lift phase or trajectory, and does that benefit hold across people and physical conditions?** This is a testable benchmark question, not a verified novel method. WAY can provide movement targets; choose either phase detection or limited trajectory prediction for the first pilot, not all targets at once.

Mario's robot policy would execute a physically meaningful task with contact; measured human motion can supply targets or demonstrations, requiring an explicit human-to-robot retargeting map. Compare EEG-conditioned inputs with cue-only, previous-motion-only, shuffled-EEG and oracle inputs under matched episodes and budgets. Keep force, glove, true phase labels and future samples out of EEG-only policy inputs. This distinguishes actual information from a demonstration that merely animates the recorded task schedule.

For a movement-conditioned EEG generator, train using paired *human* recordings and evaluate augmentation on held-out *real* EEG. Robot trajectories alone do not establish authentic human EEG. Likewise, original recordings cannot establish live adaptation to a new robot's behavior. Retargeting and source-to-new-robot feedback are distinct assumptions.

## Small pilot before committing

1. Freeze one person's single series as a format/alignment pilot; reserve other series/people for later evaluation. Download no full corpus yet.
2. List channels, units, original timestamps, missing values and event codes. Preserve original values before any loader's zero filling. Plot EEG, motion and forces against the same events and verify clock alignment.
3. Estimate actual usable paired-trial count, storage and preparation cost. Reject a candidate if required targets are missing or timestamps cannot be validated.
4. Compare cue-time-only and prior-motion baselines with a small causal EEG decoder. Split whole trials/runs before extracting windows; fit preprocessing and thresholds inside training only. Use shuffled-label/time-shift controls.
5. If there is measurable information beyond those shortcuts, port one bounded target to the robot simulator. Otherwise reassess the question before investing TACC training time.

No current finalist supports universal decoding across arbitrary EEG, authentic EEG reconstruction from arbitrary robot motion, or automatic zero-shot recognition of unseen movement semantics. Independent datasets must share an explicitly matched target; similarity of words such as 'grasp' is insufficient.
