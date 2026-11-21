"""Ensure tests always import this checkout's packages ahead of shared .venv .pth paths."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SIM_SRC = REPO_ROOT / "simulation" / "src"

for candidate in (SIM_SRC, REPO_ROOT):
    path_str = str(candidate)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

for mod_name in list(sys.modules):
    if mod_name == "fri_bci_sim" or mod_name.startswith("fri_bci_sim.") or mod_name == "eeg_decoder" or mod_name.startswith("eeg_decoder."):
        mod = sys.modules.get(mod_name)
        mod_file = getattr(mod, "__file__", None)
        if mod_file and not Path(mod_file).resolve().is_relative_to(REPO_ROOT):
            del sys.modules[mod_name]
