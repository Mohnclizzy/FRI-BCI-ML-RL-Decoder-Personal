"""Acquire the fixed NVIDIA artifact and patch a NEW candidate-only working copy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://developer.download.nvidia.com/Omniverse/learning/Courses/GettingStartedNewton/getting_started_with_newton_franka_cube.zip"
ARCHIVE_SHA256 = "d0fdef74d06752b224e78402437de2b8aaf5ee76145f13338bd81049a9f75f67"
CHECKPOINT_SHA256 = "832d8262ede665b5698c557e79ef0aa4167ec0fae1b5d5d8b222e829effc24a4"
ISAAC_COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(archive: Path, work: Path):
    if work.exists():
        raise FileExistsError(f"Refusing to overwrite {work}")
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive hash mismatch")
    with zipfile.ZipFile(archive) as z:
        # Validate the entire archive before writing; never execute shipped pyc.
        for entry in z.infolist():
            target = (work / entry.filename).resolve()
            if not target.is_relative_to(work.resolve()) or entry.filename.startswith("/"):
                raise ValueError("Unsafe archive path")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlink is not allowed")
        work.mkdir(parents=True)
        for entry in z.infolist():
            if "__pycache__" not in Path(entry.filename).parts and not entry.filename.endswith(".pyc"):
                z.extract(entry, work)
    root = work / "franka_cube"
    checkpoint = root / "checkpoint/model_399.pt"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint hash mismatch")
    cfg = root / "source/franka_cube/franka_cube/tasks/direct/franka_cube/franka_cube_env_cfg.py"
    original = cfg.read_text(encoding="utf-8")
    patched = original
    changes = [
        ("from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG",
         "from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_CFG as FRANKA_PANDA_CFG"),
        (".velocity_limit_sim =", ".joint_velocity_limit ="),
        (".effort_limit_sim =", ".joint_effort_limit ="),
    ]
    patch_log = []
    for before, after in changes:
        count = patched.count(before)
        if not count:
            raise ValueError(f"Unexpected upstream source: missing {before}")
        patched = patched.replace(before, after)
        patch_log.append({"before": before, "after": after, "count": count})
    cfg.write_text(patched, encoding="utf-8")
    env_source = root / "source/franka_cube/franka_cube/tasks/direct/franka_cube/franka_cube_env.py"
    env_original = env_source.read_text(encoding="utf-8")
    old_clone = "self.scene.clone_environments(copy_from_source=False)"
    if env_original.count(old_clone) != 1:
        raise ValueError("Unexpected scene cloning source")
    env_source.write_text(env_original.replace(old_clone,
        "# Pinned DirectRLEnv registers the scene and performs cloning after _setup_scene."), encoding="utf-8")
    patch_log.append({"before": old_clone, "after": "cloning owned by pinned DirectRLEnv", "count": 1})
    manifest = {
        "source_url": URL, "archive_sha256": sha256(archive),
        "checkpoint_sha256": sha256(checkpoint), "target_isaac_commit": ISAAC_COMMIT,
        "patches": patch_log, "config_before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "config_after_sha256": sha256(cfg), "root": str(root.resolve()),
        "env_before_sha256": hashlib.sha256(env_original.encode()).hexdigest(),
        "env_after_sha256": sha256(env_source),
        "condition": "nvidia_pretrained_candidate_compat_v1",
        "physics_validation": "not_yet_run", "original_archive_preserved": True,
    }
    (work / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive destination prevents accidental replacement of a previous download.
        with urllib.request.urlopen(URL, timeout=60) as response, args.archive.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    print(json.dumps(prepare(args.archive, args.work), indent=2))


if __name__ == "__main__":
    main()
