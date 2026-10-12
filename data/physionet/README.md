# Shared baseline EEG recordings

This directory contains the exact 15 original EDF+ files used by the project's
first baseline: PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB), version
1.0.0, participants S001-S005 and imagery runs R04, R08, R12. This is a selected
subset, not the full PhysioNet dataset. The original recordings are unmodified.

Source: [Gerwin Schalk's dataset on PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/).
License and attribution: [ATTRIBUTION.md](ATTRIBUTION.md).
Exact source URLs, file sizes, and SHA-256 checksums: [manifest.json](manifest.json).

## Recording layout

The EDF files are in `MNE-eegbci-data/files/eegmmidb/1.0.0/`, under folders
`S001` through `S005`. For example:

```text
data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf
```

This matches the existing MNE loader's cache layout and `configs/baseline.json`,
so the baseline can reuse the committed recordings after a clone. Other data
downloads remain ignored by Git. These 15 files total 38,581,920 bytes
(approximately 36.8 MiB); each individual file is approximately 2.6 MB.

## Labels and baseline use

Each file contains 64 EEG channels sampled at 160 Hz, with task annotations
embedded in the EDF+ file. Separate `.event` sidecar files are not required by
our loader and are not included.

| Annotation in these runs | Instruction | Binary decoder class |
|---|---|---|
| T0 | Rest | REST |
| T1 | Imagine opening and closing the left fist | HAND_IMAGERY |
| T2 | Imagine opening and closing the right fist | HAND_IMAGERY |

Runs 4 and 8 supply training windows; run 12 supplies test windows from the same
five people. The decoder selects nine electrodes and extracts one two-second
window per valid annotation, starting one second after onset. The baseline
produced 450 windows, balanced between REST and HAND_IMAGERY. These prepared
arrays are generated locally rather than included with the raw recordings.

The annotations identify instructed imagery, not independently verified thoughts.
They do not distinguish separate hand opening, closing, wrist flexion, or twist.

## Verify the recordings

From the repository root, run this dependency-free check:

```powershell
python scripts/verify_shared_data.py
```

It checks every listed file's size and SHA-256 against the manifest. The recorded
hashes were also verified against the MNE EEGBCI checksum registry when this
subset was added. Preserve the original files and place derived data in `outputs/`.

## Open EEG waveforms

GitHub stores EDF recordings as binary downloads and does not display an EEG
waveform preview. Clone this branch, install the requirements from the project
README, and use MNE in a Python console or notebook from the repository root:

```python
from pathlib import Path
import mne

path = Path("data/physionet/MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R04.edf")
raw = mne.io.read_raw_edf(path, preload=True)
print(raw.info)
print(raw.annotations)
mne.viz.set_browser_backend("matplotlib")
raw.plot(n_channels=12, duration=10, block=True)
```

The interactive plot requires a desktop environment with a working Matplotlib
GUI backend. `print(raw.info)` and `print(raw.annotations)` also work without a GUI.
The displayed data are the original signals, before the decoder's preprocessing.
