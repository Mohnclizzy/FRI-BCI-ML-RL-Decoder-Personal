# Literature Reference Summary for EEG Intent Decoding and Shared Robotic-Hand Control

Generated from the uploaded `literature_review.tex` on 2026-10-12. This document maps every distinct citation key in the TeX file to its verified source, summarizes the technical contribution, preserves its relevance to the EEG-to-robotic-hand simulation project, and notes limitations or interpretation risks.

## Executive Synthesis

The cited literature supports a conservative, simulation-first architecture: decode a small, experiment-defined intent vocabulary from public EEG; temporally filter the decoder output; then let a robot-control policy handle low-level movement. The strongest methodological baselines are transparent EEG pipelines such as CSP/FBCSP, Riemannian covariance features, and compact CNNs such as EEGNet. The robot-control literature supports shared autonomy and RL in simulation, but it does not justify treating offline EEG classification accuracy as proof of reliable online robotic-hand control.

The review's main boundary condition is public-data replay. Public datasets can test offline decoding and command-stream robustness, but they cannot validate user adaptation, closed-loop feedback learning, clinical benefit, or physical hand safety. Those require live-user and hardware-specific studies.

## Distinct Citation Keys Identified

| Citation key | Verified source | Primary project role | Verification note |
|---|---|---|---|
| `akinola2017` | Akinola et al., "Task Level Hierarchical System for BCI-Enabled Shared Autonomy" | Shared-autonomy precedent for low-bandwidth BCI commands | Verified DOI and PDF |
| `ang2008` | Ang et al., "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface" | Classical motor-imagery feature baseline | Verified DOI and proceedings metadata |
| `baniqued2021` | Baniqued et al., "Brain-Computer Interface Robotics for Hand Rehabilitation After Stroke" | Evidence on hand-robot BCI systems and their limits | Verified DOI, open-access article |
| `barachant2012` | Barachant et al., "Multiclass Brain-Computer Interface Classification by Riemannian Geometry" | Riemannian covariance classifier baseline | Verified DOI, PubMed/HAL metadata |
| `blankertz2008` | Blankertz et al., "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis" | CSP theory and single-trial EEG baseline | Verified DOI and journal metadata |
| `brunke2022` | Brunke et al., "Safe Learning in Robotics" | Safety framing for RL and learned control | Verified DOI/arXiv preprint |
| `gramfort2013` | Gramfort et al., "MEG and EEG Data Analysis with MNE-Python" | Reproducible EEG preprocessing stack | Verified DOI and Frontiers article |
| `isaaclab` | NVIDIA Isaac Lab documentation | Simulation and robot-learning infrastructure | Verified current documentation URL |
| `jayaram2018` | Jayaram and Barachant, "MOABB: Trustworthy Algorithm Benchmarking for BCIs" | Cross-dataset benchmarking and reproducibility | Original TeX DOI is incorrect; verified DOI is `10.1088/1741-2552/aadea0` |
| `lawhern2018` | Lawhern et al., "EEGNet: A Compact Convolutional Neural Network..." | Compact deep-learning baseline | Verified DOI, PubMed/arXiv |
| `lottejeunet2018` | Lotte and Jeunet, "Defining and Quantifying Users' Mental Imagery-Based BCI Skills" | User variability and BCI skill metrics | Verified DOI/PubMed |
| `lotte2018` | Lotte et al., "A Review of Classification Algorithms..." | Comparative ML guidance for EEG-BCI | Verified DOI/IOP |
| `mcfarland2017` | McFarland and Wolpaw, "EEG-Based Brain-Computer Interfaces" | Contemporary overview of EEG-BCI limitations | Verified DOI/PubMed |
| `schalk2009` | Schalk, "EEG Motor Movement/Imagery Dataset" | Public EEG data source | Verified PhysioNet record and DOI |
| `schirrmeister2017` | Schirrmeister et al., "Deep Learning with CNNs for EEG Decoding and Visualization" | Deep ConvNet comparison and interpretability | Verified DOI/Wiley/arXiv |
| `rslrl` | Schwarke et al., "RSL-RL: A Learning Library for Robotics Research" | RL implementation library for robotics | Verified arXiv and GitHub link |
| `wolpaw2002` | Wolpaw et al., "Brain-computer interfaces for communication and control" | Foundational BCI framing | Verified DOI/PubMed |
| `zhang2021` | Zhang and Wang, "A Survey on Robots Controlled by Motor Imagery BCIs" | Survey of MI-BCI robot applications | Verified DOI/ScienceDirect |

## Citation-to-Source Mapping in the Literature Review

The TeX file uses the cited sources in four main ways:

- Decoder baselines and algorithms: `blankertz2008`, `ang2008`, `barachant2012`, `lawhern2018`, `schirrmeister2017`, `lotte2018`
- BCI problem setting and robot translation: `wolpaw2002`, `mcfarland2017`, `baniqued2021`, `zhang2021`, `akinola2017`
- Public data and reproducibility: `schalk2009`, `gramfort2013`, `jayaram2018`, `lottejeunet2018`
- Robot-learning infrastructure and safety framing: `isaaclab`, `rslrl`, `brunke2022`

Several citations are intentionally reused across sections because the proposed system is a pipeline: data assumptions constrain decoders, decoder uncertainty constrains robot commands, and robot embodiment constrains what the command can safely mean.

## In-Depth Source Summaries

### `wolpaw2002` — Wolpaw et al., "Brain-computer interfaces for communication and control"

Verified record: Clinical Neurophysiology, 113(6), 767-791, DOI `10.1016/S1388-2457(02)00057-3`. PubMed verifies the article and abstracts its core argument: BCI progress depends on interdisciplinary research, reliable signal selection, translation algorithms, artifact control, objective evaluation, long-term assessment, appropriate applications, and user acceptance.

Technical contribution: This is a foundational review defining BCIs as systems that translate brain activity into communication or control without relying on normal motor output. It emphasizes that BCI design is not only a classification problem; it includes signal acquisition, feature extraction, translation, feedback, training, user acceptance, and objective performance measurement.

Findings and methods: The paper synthesizes evidence across invasive and non-invasive BCI modalities, including EEG rhythms and evoked potentials. Its strongest project-relevant message is that BCI signals are low-bandwidth, noisy, and user-dependent, so system evaluation must go beyond one offline metric.

Limitations for this project: The review is broad and foundational rather than specific to robotic hands or modern deep learning. It should justify cautious BCI framing, not any specific decoder implementation.

Project relevance: It supports the literature review's claim that the proposed system should be described as intent decoding plus command translation, not as direct low-level physical-hand control.

### `mcfarland2017` — McFarland and Wolpaw, "EEG-Based Brain-Computer Interfaces"

Verified record: Current Opinion in Biomedical Engineering, 4, 194-200, DOI `10.1016/j.cobme.2017.11.004`; PubMed lists PMCID `PMC5839510`.

Technical contribution: This article updates the EEG-BCI picture for modern systems. It defines BCIs as real-time computer systems that translate brain signals into useful commands and stresses that many systems remain proof-of-principle rather than widely deployed technology.

Findings and methods: The review identifies sensor quality, stable signals, real-time signal processing, and mutual adaptation between user and BCI as major barriers. It also emphasizes rehabilitation as a promising but still demanding application area.

Limitations for this project: It does not validate any specific public dataset or robotic-hand controller. Its value is conceptual: it warns that offline classifier evaluation is not equivalent to usable online control.

Project relevance: This source supports using replay as a limited experimental tool while explicitly labeling it as simulated acquisition, not live BCI validation.

### `blankertz2008` — Blankertz et al., "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis"

Verified record: IEEE Signal Processing Magazine, 25(1), 41-56, DOI `10.1109/MSP.2008.4408441`.

Technical contribution: The paper explains common spatial patterns (CSP), a supervised spatial-filtering method for multichannel EEG. CSP learns filters that maximize variance for one condition while minimizing variance for another, often after band-pass filtering around rhythms affected by event-related desynchronization/synchronization.

Findings and methods: CSP is presented as a practical way to improve signal-to-noise ratio for single-trial EEG classification. It is especially relevant for two-condition motor-imagery problems because it produces spatial filters that can be inspected and related back to sensorimotor regions.

Limitations for this project: CSP assumes well-defined classes and can be sensitive to nonstationarity, small samples, and train/test leakage. It also does not solve command stability; it only produces features for classification.

Project relevance: CSP+LDA should be a primary transparent baseline for rest-versus-movement and possibly binary movement distinctions. Its interpretability makes it useful before testing deep models.

### `ang2008` — Ang et al., "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface"

Verified record: IJCNN 2008, pages 2390-2397, DOI `10.1109/IJCNN.2008.4634130`.

Technical contribution: FBCSP extends CSP by applying CSP over multiple frequency bands, then selecting discriminative band/spatial features for classification. This directly addresses uncertainty about which frequency band best separates motor-imagery classes.

Findings and methods: The approach decomposes EEG into a bank of frequency ranges, computes CSP features per band, then uses feature selection and classification. It is widely used in motor-imagery benchmarks and is still a strong baseline for oscillatory EEG control signals.

Limitations for this project: FBCSP has multiple preprocessing and feature-selection choices, so evaluation must keep train/test separation clean. Feature selection must occur inside cross-validation or training folds to avoid optimistic results.

Project relevance: FBCSP is a strong baseline when the project moves beyond simple CSP, especially if the chosen public data contain sensorimotor rhythm labels.

### `barachant2012` — Barachant et al., "Multiclass Brain-Computer Interface Classification by Riemannian Geometry"

Verified record: IEEE Transactions on Biomedical Engineering, 59(4), 920-928, DOI `10.1109/TBME.2011.2172210`.

Technical contribution: The paper represents each EEG trial by a spatial covariance matrix and classifies those matrices using Riemannian geometry on the manifold of symmetric positive-definite matrices. This avoids explicit CSP-style spatial filtering and treats covariance structure as the core descriptor.

Findings and methods: The paper compares minimum-distance-to-Riemannian-mean and tangent-space LDA approaches against multiclass CSP+LDA on BCI Competition IV dataset IIa. The tangent-space method improved mean classification accuracy in that study, making Riemannian tangent-space features a credible baseline.

Limitations for this project: Riemannian methods still depend on clean epoching, covariance estimation, and appropriate evaluation splits. Good offline performance does not guarantee stable command streams.

Project relevance: Riemannian covariance classification should be evaluated beside CSP/FBCSP because it often performs well with limited EEG data and has a principled representation for multichannel trials.

### `lotte2018` — Lotte et al., "A Review of Classification Algorithms for EEG-Based Brain-Computer Interfaces: A 10 Year Update"

Verified record: Journal of Neural Engineering, 15(3), 031005, DOI `10.1088/1741-2552/aab2f2`.

Technical contribution: This review surveys EEG-BCI classification work from roughly 2007-2017, organizing progress around adaptive classifiers, matrix/tensor classifiers, transfer learning, deep learning, and additional methods such as shrinkage LDA and random forests.

Findings and methods: The review confirms that no single classifier family dominates every EEG-BCI setting. It highlights the continuing importance of linear classifiers, adaptive methods, transfer learning, and classifiers suited to nonstationary EEG.

Limitations for this project: Because it is a broad review, it provides guidance rather than a ready-made pipeline. Its conclusions depend on the diversity of tasks and datasets in the surveyed literature.

Project relevance: It supports comparing multiple families under identical preprocessing and splits instead of treating deep learning as automatically superior.

### `lawhern2018` — Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces"

Verified record: Journal of Neural Engineering, 15(5), 056013, DOI `10.1088/1741-2552/aace8c`; PubMed PMID `29932424`.

Technical contribution: EEGNet is a compact CNN architecture for EEG classification. It uses depthwise and separable convolutions to encode EEG-relevant temporal and spatial filtering while keeping parameter count low.

Findings and methods: The authors test EEGNet across P300, error-related negativity, movement-related cortical potential, and sensory motor rhythm paradigms. They report that EEGNet can generalize across paradigms and remain interpretable through learned-feature visualization, especially when training data are limited.

Limitations for this project: EEGNet still needs careful validation under subject-wise splits and public-dataset heterogeneity. The original work's cross-paradigm strength does not mean it will outperform simpler baselines on every motor-imagery replay setup.

Project relevance: EEGNet is the most defensible compact deep-learning baseline. It should be compared against CSP/FBCSP and Riemannian methods under the exact same train/test protocol.

### `schirrmeister2017` — Schirrmeister et al., "Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization"

Verified record: Human Brain Mapping, 38(11), 5391-5420, DOI `10.1002/hbm.23730`.

Technical contribution: This work tests deep and shallow ConvNets for EEG decoding and shows how network visualization can reveal frequency and spatial features used by the model. The shallow ConvNet is explicitly inspired by FBCSP-like bandpower processing.

Findings and methods: The study compares ConvNet architectures with strong classical baselines and emphasizes design choices such as batch normalization, dropout, ELU activations, and spatial/temporal factorization. The authors show that deep networks can be competitive when properly designed and interpreted.

Limitations for this project: Deep ConvNets are data-hungry and can overfit EEG. Interpretability helps, but it does not eliminate the need for held-out-subject testing and command-stream evaluation.

Project relevance: This source supports testing deep models as serious candidates while keeping them bounded by baseline comparisons and visualization checks.

### `schalk2009` — Schalk, "EEG Motor Movement/Imagery Dataset"

Verified record: PhysioNet, version 1.0.0, DOI `10.13026/C28G6P`.

Technical contribution: This public dataset contains 64-channel EEG recordings from subjects performing motor and motor-imagery tasks. PhysioNet describes it as useful for motor-imagery classification, executed-versus-imagined movement comparisons, rest-versus-task analysis, and evaluation of signal-processing methods for EEG-BCI.

Findings and methods: The dataset is a resource rather than a hypothesis-testing paper. Its value is accessibility and standardized files suitable for reproducible experiments.

Limitations for this project: Labels are determined by the experiment protocol. Unsupported distinctions, such as inferring fine-grained hand open/close phases when the dataset does not label them separately, should not be invented. Replay cannot capture user adaptation to feedback.

Project relevance: This is the likely data backbone for the project. It supports rest/task and protocol-defined motor/motor-imagery experiments, not arbitrary dexterous intent labels.

### `gramfort2013` — Gramfort et al., "MEG and EEG Data Analysis with MNE-Python"

Verified record: Frontiers in Neuroscience, 7:267, DOI `10.3389/fnins.2013.00267`.

Technical contribution: MNE-Python is an open-source Python ecosystem for EEG/MEG preprocessing, epoching, visualization, time-frequency analysis, ICA, statistics, and decoding. The paper frames MNE-Python as a way to standardize M/EEG analysis pipelines and improve reproducibility.

Findings and methods: The paper describes the standard analysis workflow using raw, epochs, and evoked data structures, along with filtering, event extraction, covariance estimation, and visualization tools. It integrates with NumPy, SciPy, matplotlib, and scikit-learn.

Limitations for this project: MNE-Python is infrastructure, not evidence that a particular decoder generalizes. Reproducible software still requires correct experimental design.

Project relevance: MNE should be used to load, preprocess, epoch, and audit the EEG data so results are traceable and reproducible.

### `jayaram2018` — Jayaram and Barachant, "MOABB: Trustworthy Algorithm Benchmarking for BCIs"

Verified record: Journal of Neural Engineering, 15(6), 066011, verified DOI `10.1088/1741-2552/aadea0`.

Correction: The uploaded TeX bibliography lists `10.1088/1741-2552/aae5d3`, but the MOABB GitHub citation metadata and article records identify `10.1088/1741-2552/aadea0`.

Technical contribution: MOABB provides an open-source benchmarking framework for BCI algorithms across open datasets. It builds on MNE and scikit-learn and was designed to make BCI comparisons more trustworthy and reproducible.

Findings and methods: The authors evaluate multiple decoding pipelines across 12 open datasets and more than 250 subjects. Their key finding is that identical pipelines can rank differently across datasets, so a single favorable dataset or split is not enough evidence for generality.

Limitations for this project: MOABB is a benchmarking framework, not a direct robot-control evaluation. It helps evaluate decoders, but command filtering and robot behavior still need separate tests.

Project relevance: The project should report per-subject and cross-subject metrics, use group-aware splits, and avoid conclusions based only on pooled accuracy.

### `lottejeunet2018` — Lotte and Jeunet, "Defining and Quantifying Users' Mental Imagery-Based BCI Skills: A First Step"

Verified record: Journal of Neural Engineering, 15(4), 046030, DOI `10.1088/1741-2552/aac577`.

Technical contribution: This paper separates classifier performance from user skill in mental-imagery BCI. It argues that classification accuracy alone is a poor metric for understanding whether a user can self-modulate EEG patterns.

Findings and methods: The authors propose metrics for MI-BCI skill based on the user's ability to modulate EEG features, not only the classifier's output. Reanalysis shows that classifier accuracy can hide both skill improvement and failure to modulate the intended pattern.

Limitations for this project: Public-data replay cannot measure how a participant learns from feedback during the proposed robotic-hand task. It can only analyze recorded signals and simulated command streams.

Project relevance: This source supports explicitly separating decoder performance from user-control ability and avoiding claims about live BCI usability from offline replay alone.

### `baniqued2021` — Baniqued et al., "Brain-Computer Interface Robotics for Hand Rehabilitation After Stroke: A Systematic Review"

Verified record: Journal of NeuroEngineering and Rehabilitation, 18:15, DOI `10.1186/s12984-021-00820-8`.

Technical contribution: This systematic review examines EEG-based BCI robotic systems for hand rehabilitation after stroke, including technical systems and clinical evidence.

Findings and methods: The review searched multiple databases for January 2010 through October 2019. It identified 30 eligible studies; 11 involved chronic or subacute stroke patients. A key project-relevant finding is that most systems used motor imagery to trigger grasping or pinching movements rather than continuously controlling dexterous joints. The review also found heterogeneous reporting and a need for standardized technical and clinical protocols.

Limitations for this project: The clinical rehabilitation setting differs from a simulation-first robotic-hand controller. The review cannot be used to claim clinical efficacy for the current project.

Project relevance: It supports a discrete command vocabulary and warns against overstating dexterous, continuous EEG hand control.

### `zhang2021` — Zhang and Wang, "A Survey on Robots Controlled by Motor Imagery Brain-Computer Interfaces"

Verified record: Cognitive Robotics, 1, 12-24, DOI `10.1016/j.cogr.2021.02.001`.

Technical contribution: This survey reviews robots controlled by motor-imagery BCIs across EEG paradigms, signal-processing methods, and robot applications, including rehabilitation robots, wheelchairs, nursing beds, and unmanned vehicles.

Findings and methods: The paper organizes MI-BCI robot systems by evocation paradigm, feature extraction/classification approach, and application scenario. Its major conclusion is that future systems need better robustness, error correction, adaptive learning, and control accuracy.

Limitations for this project: It is a survey rather than a controlled comparison. It does not validate the proposed decoder-filter-policy pipeline directly.

Project relevance: It supports designing explicit error handling and command filtering as first-class components, not as optional polish.

### `akinola2017` — Akinola et al., "Task Level Hierarchical System for BCI-Enabled Shared Autonomy"

Verified record: IEEE-RAS Humanoids 2017, pages 219-225, DOI `10.1109/HUMANOIDS.2017.8246878`.

Technical contribution: This paper describes a hierarchical shared-autonomy system where a low-bandwidth EEG BCI using SSVEP lets a user provide high-level selections while a humanoid robot handles lower-level task execution.

Findings and methods: The system asks the user for high-level choices only when needed, reducing the burden of continuous BCI control. The paper reports testing in simulation and physical settings, emphasizing a division between user intent and autonomous robot operation.

Limitations for this project: The signal paradigm is SSVEP, not motor imagery, and the robot embodiment is a humanoid rather than a robotic hand. Therefore it is an architectural precedent, not a direct decoder comparison.

Project relevance: It strongly supports the proposed design choice: EEG should select an intent or primitive, while the robot policy handles low-level control.

### `brunke2022` — Brunke et al., "Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning"

Verified record: Annual Review of Control, Robotics, and Autonomous Systems, 5, 411-444, DOI `10.1146/annurev-control-042920-020211`; arXiv preprint `2108.06266`.

Technical contribution: This review organizes safe learning in robotics across learning-based control, safe reinforcement learning, uncertainty modeling, constraints, and formal certification.

Findings and methods: It distinguishes methods that encourage robustness from methods that formally enforce constraints or certify safety. It also emphasizes realistic physics-based benchmarks for fair comparison.

Limitations for this project: The proposed EEG-to-hand project can implement command filters, constraints, and simulation stress tests, but that does not equal formal safety certification.

Project relevance: It gives the language needed to avoid overclaiming "safe RL." The project should say "risk-aware command filtering and constraint-aware simulation evaluation" unless formal guarantees are actually implemented.

### `isaaclab` — NVIDIA, "Isaac Lab Documentation: A Unified Framework for Robot Learning"

Verified record: Official Isaac Lab documentation, accessed 2026-10-12.

Technical contribution: Isaac Lab is a modular robot-learning framework built on NVIDIA Isaac Sim. Its documentation describes support for reinforcement learning, learning from demonstrations, motion planning, fast physics simulation, vectorized rendering, domain randomization, and ready-made environments.

Findings and methods: Isaac Lab is infrastructure for building and evaluating robot-learning environments. It is appropriate for simulation-first control experiments, especially where repeated trials, injected errors, and robot-state logging are needed.

Limitations for this project: Documentation is not a peer-reviewed empirical claim that a trained hand policy will transfer to hardware. Isaac Lab supports simulation and policy training; it does not validate EEG intent decoding.

Project relevance: It is the correct infrastructure citation for the simulated hand environment and policy-training harness.

### `rslrl` — Schwarke et al., "RSL-RL: A Learning Library for Robotics Research"

Verified record: arXiv `2509.10771`, "RSL-RL: A Learning Library for Robotics Research"; GitHub `leggedrobotics/rsl_rl`.

Technical contribution: RSL-RL is an open-source reinforcement-learning library tailored to robotics. Its design emphasizes a compact, modifiable codebase; GPU-only high-throughput training; and robotics-oriented components such as runners, algorithms, and network modules.

Findings and methods: The paper states that RSL-RL is validated in simulation benchmarks and real-world robotic experiments, with a focus on making learning-based robotic controllers practical to develop.

Limitations for this project: RSL-RL is an implementation tool, not evidence that an EEG-conditioned hand policy will be reliable. The task environment, rewards, safety constraints, and command-noise tests determine whether the controller is meaningful.

Project relevance: It is a reasonable starting library for training Isaac Lab policies once the hand task and command interface are defined.

## Cross-Source Methodological Guidance

### Decoder Design

Use at least three decoder families:

1. CSP or FBCSP with LDA/shrinkage LDA.
2. Riemannian covariance features with minimum-distance or tangent-space classification.
3. EEGNet or a similarly compact CNN.

All preprocessing, feature selection, normalization, and hyperparameter tuning must occur inside training folds. Metrics should include balanced accuracy, macro-F1, confusion matrices, per-subject performance, and held-out-subject results where feasible.

### Command Filtering

The literature does not support mapping every EEG window directly to robotic-hand joint actions. A defensible system should use a command filter with rest priority, confidence thresholds, dwell time, majority voting or persistence, and explicit rejection/hold states. Evaluate accepted-command rate, false activation rate, switch rate, and latency.

### Robot-Control Evaluation

RL should act as the low-level controller conditioned on an accepted high-level intent. Evaluate simulated task success, unintended transitions, joint/velocity limit violations, resets, collision events, and robustness under injected command errors. A stricter filter should be expected to reduce false activations while increasing latency; this tradeoff should be measured rather than assumed.

### Data and Label Discipline

Labels must come from the dataset protocol. Public EEG data can support rest-versus-task, executed-versus-imagined movement, and protocol-defined motor imagery labels, but it cannot justify arbitrary fine-grained hand-command labels unless those labels exist in the dataset.

## Verified Bibliography and Links

- Akinola, I., Chen, B., Koss, J., Patankar, A., Varley, J., & Allen, P. (2017). "Task Level Hierarchical System for BCI-Enabled Shared Autonomy." IEEE-RAS Humanoids, 219-225. DOI: `10.1109/HUMANOIDS.2017.8246878`. https://www.cs.columbia.edu/~allen/PAPERS/Humanoids_2017.pdf
- Ang, K. K., Chin, Z. Y., Zhang, H., & Guan, C. (2008). "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface." IJCNN, 2390-2397. DOI: `10.1109/IJCNN.2008.4634130`. https://doi.org/10.1109/IJCNN.2008.4634130
- Baniqued, P. D. E., et al. (2021). "Brain-Computer Interface Robotics for Hand Rehabilitation After Stroke: A Systematic Review." Journal of NeuroEngineering and Rehabilitation, 18:15. DOI: `10.1186/s12984-021-00820-8`. https://pmc.ncbi.nlm.nih.gov/articles/PMC7825186/
- Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). "Multiclass Brain-Computer Interface Classification by Riemannian Geometry." IEEE Transactions on Biomedical Engineering, 59(4), 920-928. DOI: `10.1109/TBME.2011.2172210`. https://pubmed.ncbi.nlm.nih.gov/22010143/
- Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Muller, K. R. (2008). "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis." IEEE Signal Processing Magazine, 25(1), 41-56. DOI: `10.1109/MSP.2008.4408441`. https://doi.org/10.1109/MSP.2008.4408441
- Brunke, L., et al. (2022). "Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning." Annual Review of Control, Robotics, and Autonomous Systems, 5, 411-444. DOI: `10.1146/annurev-control-042920-020211`. https://arxiv.org/abs/2108.06266
- Gramfort, A., et al. (2013). "MEG and EEG Data Analysis with MNE-Python." Frontiers in Neuroscience, 7:267. DOI: `10.3389/fnins.2013.00267`. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2013.00267/full
- NVIDIA. (2026). "Isaac Lab Documentation: A Unified Framework for Robot Learning." https://isaac-sim.github.io/IsaacLab/main/index.html
- Jayaram, V., & Barachant, A. (2018). "MOABB: Trustworthy Algorithm Benchmarking for BCIs." Journal of Neural Engineering, 15(6), 066011. Correct DOI: `10.1088/1741-2552/aadea0`. https://arxiv.org/abs/1805.06427
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces." Journal of Neural Engineering, 15(5), 056013. DOI: `10.1088/1741-2552/aace8c`. https://pubmed.ncbi.nlm.nih.gov/29932424/
- Lotte, F., & Jeunet, C. (2018). "Defining and Quantifying Users' Mental Imagery-Based BCI Skills: A First Step." Journal of Neural Engineering, 15(4), 046030. DOI: `10.1088/1741-2552/aac577`. https://pubmed.ncbi.nlm.nih.gov/29769435/
- Lotte, F., et al. (2018). "A Review of Classification Algorithms for EEG-Based Brain-Computer Interfaces: A 10 Year Update." Journal of Neural Engineering, 15(3), 031005. DOI: `10.1088/1741-2552/aab2f2`. https://doi.org/10.1088/1741-2552/aab2f2
- McFarland, D. J., & Wolpaw, J. R. (2017). "EEG-Based Brain-Computer Interfaces." Current Opinion in Biomedical Engineering, 4, 194-200. DOI: `10.1016/j.cobme.2017.11.004`. https://pubmed.ncbi.nlm.nih.gov/29527584/
- Schalk, G. (2009). "EEG Motor Movement/Imagery Dataset" version 1.0.0. PhysioNet. DOI: `10.13026/C28G6P`. https://physionet.org/content/eegmmidb/1.0.0/
- Schirrmeister, R. T., et al. (2017). "Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization." Human Brain Mapping, 38(11), 5391-5420. DOI: `10.1002/hbm.23730`. https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23730
- Schwarke, C., Mittal, M., Rudin, N., Hoeller, D., & Hutter, M. (2025). "RSL-RL: A Learning Library for Robotics Research." arXiv:2509.10771. https://arxiv.org/abs/2509.10771
- Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M. (2002). "Brain-computer interfaces for communication and control." Clinical Neurophysiology, 113(6), 767-791. DOI: `10.1016/S1388-2457(02)00057-3`. https://pubmed.ncbi.nlm.nih.gov/12048038/
- Zhang, J., & Wang, M. (2021). "A Survey on Robots Controlled by Motor Imagery Brain-Computer Interfaces." Cognitive Robotics, 1, 12-24. DOI: `10.1016/j.cogr.2021.02.001`. https://www.sciencedirect.com/science/article/pii/S266724132100001X

