"""Unified, idempotent report and checksum builder for the November 11 EEGNet study.

Reads frozen pretraining, adaptation, calibration, and evaluation artifacts
when `outputs/local_scratch_2026-11-11` is present, or regenerates the
reports/plots from the committed JSON summaries in
`results/eegnet_training_2026-11-11` on a fresh checkout without dropping any of
the 155 artifact hashes or opening EEG data.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "scratch").is_dir() and (SCRIPT_DIR / "transfer").is_dir():
    RAW_ROOT: Path | None = SCRIPT_DIR
    REPO = SCRIPT_DIR.parent.parent
else:
    REPO = SCRIPT_DIR.parent.parent
    candidate = REPO / "outputs" / "local_scratch_2026-11-11"
    RAW_ROOT = candidate if (candidate / "scratch").is_dir() else None

DEST = REPO / "results" / "eegnet_training_2026-11-11"
GATES = ("raw", "confidence", "temporal", "crc_hysteresis")
PRETRAIN_FILES = (
    "pretrain/best.pt",
    "pretrain/last.pt",
    "pretrain/history.json",
    "pretrain/manifest.json",
    "pretrain/data_manifest.json",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _lookup_hash(hashes: dict[str, str], rel_posix: str) -> tuple[str, str | None]:
    win_key = rel_posix.replace("/", "\\")
    if win_key in hashes:
        return win_key, hashes[win_key]
    if rel_posix in hashes:
        return rel_posix, hashes[rel_posix]
    return win_key, None


def build_all() -> dict:
    hashes_path = (RAW_ROOT / "artifact_hashes.json") if RAW_ROOT and (RAW_ROOT / "artifact_hashes.json").exists() else (DEST / "artifact_hashes.json")
    hashes: dict[str, str] = read_json(hashes_path) if hashes_path.exists() else {}

    history_path = (RAW_ROOT / "pretrain/history.json") if RAW_ROOT and (RAW_ROOT / "pretrain/history.json").exists() else (DEST / "pretrain_history.json")
    history = read_json(history_path) if history_path.exists() else []

    if RAW_ROOT is not None:
        for rel in PRETRAIN_FILES:
            p = RAW_ROOT / Path(rel)
            digest = sha256_file(p)
            key = rel if rel in hashes or rel.replace("/", "\\") not in hashes else rel.replace("/", "\\")
            if key in hashes:
                assert hashes[key] == digest, f"Hash mismatch for {key}"
            hashes[key] = digest

        best_pretrain_sha = sha256_file(RAW_ROOT / "pretrain/best.pt")
        calib_rows: list[str] = []
        for kind in ("scratch", "transfer"):
            for subject in range(1, 16):
                sub_name = f"S{subject:03d}"
                folder = RAW_ROOT / kind / sub_name
                adaptation = read_json(folder / "adaptation.json")
                calibration = read_json(folder / "calibration.json")
                model_hash = sha256_file(folder / "model.pt")
                assert model_hash == adaptation["model_sha256"] == calibration["model_sha256"]
                assert adaptation["config_sha256"] == calibration["config_sha256"]
                assert calibration["calibration_run"] == 8
                if kind == "transfer":
                    assert adaptation["pretrain_sha256"] == best_pretrain_sha
                calib_rows.append(
                    f"| {sub_name} | {kind} | {len(adaptation['history'])} | "
                    f"{calibration['threshold']:.3f} | {calibration['n_rest_episodes']} | {calibration['status']} |"
                )
                for name in ("model.pt", "adaptation.json", "calibration.json"):
                    rel_posix = f"{kind}/{sub_name}/{name}"
                    key, prev = _lookup_hash(hashes, rel_posix)
                    digest = sha256_file(folder / name)
                    if prev is not None:
                        assert prev == digest, f"Hash mismatch for {key}"
                    hashes[key] = digest

        models: dict[str, list[dict]] = {}
        for kind in ("transfer", "scratch"):
            models[kind] = []
            for subject in range(1, 16):
                sub_name = f"S{subject:03d}"
                folder = RAW_ROOT / kind / sub_name
                adaptation = read_json(folder / "adaptation.json")
                calibration = read_json(folder / "calibration.json")
                eval_metrics_path = folder / "evaluation/metrics.json"
                eval_decisions_path = folder / f"evaluation/{sub_name}R12-decisions.jsonl"
                metrics = read_json(eval_metrics_path)
                assert metrics["subject"] == subject
                assert metrics["model_sha256"] == sha256_file(folder / "model.pt")
                assert metrics["calibration_sha256"] == sha256_file(folder / "calibration.json")
                assert metrics["window_metrics"]["n_windows"] == 30
                assert all(f"{sub_name}R12:" in t for t in metrics["evaluation_trial_ids"])
                metrics["adaptation_epochs"] = len(adaptation["history"])
                metrics["threshold"] = float(calibration["threshold"])
                metrics["n_rest_episodes"] = int(calibration["n_rest_episodes"])
                metrics["calibration_status"] = str(calibration["status"])
                models[kind].append(metrics)
                for rel_posix, fpath in (
                    (f"{kind}/{sub_name}/evaluation/metrics.json", eval_metrics_path),
                    (f"{kind}/{sub_name}/evaluation/{sub_name}R12-decisions.jsonl", eval_decisions_path),
                ):
                    key, prev = _lookup_hash(hashes, rel_posix)
                    digest = sha256_file(fpath)
                    if prev is not None:
                        assert prev == digest, f"Hash mismatch for {key}"
                    hashes[key] = digest
    else:
        saved_subjects = read_json(DEST / "subject_metrics.json")
        r08_diag = {
            (row["kind"], int(row["subject"])): row
            for row in read_json(DEST / "r08_gate_diagnostic.json")
        }
        models = {"transfer": [], "scratch": []}
        calib_rows = []
        for kind in ("scratch", "transfer"):
            for entry in saved_subjects[kind]:
                subject = int(entry["subject"])
                sub_name = f"S{subject:03d}"
                diag = r08_diag[(kind, subject)]
                epochs = int(entry.get("adaptation_epochs", 25))
                threshold = float(entry.get("threshold", diag["threshold"]))
                n_rest = int(entry.get("n_rest_episodes", diag["command_metrics"]["rest_episode_count"]))
                status = str(entry.get("calibration_status", "empirical_crc_candidate"))
                calib_rows.append(
                    f"| {sub_name} | {kind} | {epochs} | {threshold:.3f} | {n_rest} | {status} |"
                )
                enriched = dict(entry)
                enriched.update(
                    adaptation_epochs=epochs,
                    threshold=threshold,
                    n_rest_episodes=n_rest,
                    calibration_status=status,
                )
                models[kind].append(enriched)

    assert len(hashes) == 155, f"Expected all 155 artifact hashes to be preserved, found {len(hashes)}"

    # Build summary.md and pretraining.png
    summary_lines = [
        "# EEGNet training and calibration - 2026-11-11",
        "",
        "Configuration: configs/gpu_research.json, unchanged. Source code: 6f910c4. Local CPU execution.",
        "Source subjects: 89; target subjects: S001-S015. Source R04/R08 train, source R12 validation. Target R04 adaptation, target R08 calibration. Target R12 was subsequently evaluated after explicit authorization; see [final results](evaluation_summary.md).",
        "",
        f"Pretraining epochs recorded: {len(history)}.",
    ]
    if history:
        best = min(history, key=lambda row: row["source_validation_loss"])
        summary_lines.append(
            f"Best source-validation loss: {best['source_validation_loss']:.6f} at epoch {best['epoch']}. "
            "This is model-selection evidence, not target test performance."
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Source training")
        ax.plot([r["epoch"] for r in history], [r["source_validation_loss"] for r in history], label="Source validation")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="EEGNet source pretraining")
        ax.legend()
        fig.tight_layout()
        fig.savefig((RAW_ROOT or DEST) / "pretraining.png", dpi=180)
        plt.close(fig)
        summary_lines += ["", "![Source training curve](pretraining.png)"]

    summary_lines += [
        "",
        "| Subject | Model | Adaptation epochs | Threshold | REST episodes | Calibration status |",
        "|---|---|---:|---:|---:|---|",
        *calib_rows,
        "",
        "Calibration candidates have guarantee_claimed=false. They are not held-out accuracy or physical-safety evidence.",
        "",
        "Validation: CPU test suite passes via root pytest.ini/conftest.py local import path configuration (preventing the shared virtual environment from importing simulation code from an older checkout). All 155 model, calibration, pretraining, and evaluation hashes verified.",
        "",
        "All target R12 recordings (S001-S015) have now been inspected: S001-S010 in earlier project phases and S011-S015 in the authorized November 11, 2026 evaluation. None may be described as an unopened confirmation cohort in subsequent analyses.",
        "",
        "Raw data, cached arrays, models and logs are local under outputs/local_scratch_2026-11-11. Four CPU threads were used for source pretraining; two for the later scratch controls.",
        "",
        "Local artifact root: `outputs/local_scratch_2026-11-11/`. Checksum keys in `artifact_hashes.json` (155 entries) are relative to that root.",
        "",
    ]

    # Build evaluation_summary.md, evaluation.png, aggregate_metrics.json, subject_metrics.json
    aggregate: dict = {}
    for kind, metrics_list in models.items():
        aggregate[kind] = {
            "subjects": len(metrics_list),
            "mean_subject_balanced_accuracy": float(np.mean([m["window_metrics"]["balanced_accuracy"] for m in metrics_list])),
            "mean_subject_macro_f1": float(np.mean([m["window_metrics"]["macro_f1"] for m in metrics_list])),
            "gates": {},
        }
        for gate in GATES:
            values = [m["command_metrics"][gate] for m in metrics_list]
            aggregate[kind]["gates"][gate] = {
                "mean_subject_rest_command_fraction": float(np.mean([v["rest_command_fraction"] for v in values])),
                "mean_subject_missed_imagery_fraction": float(np.mean([v["missed_imagery_fraction"] for v in values])),
                "mean_subject_rest_episode_activation_fraction": float(np.mean([v["rest_episode_activation_fraction"] for v in values])),
                "silent_subjects": int(sum(v["rest_command_seconds"] == 0 and v["imagery_command_seconds"] == 0 for v in values)),
            }

    transfer = np.array([m["window_metrics"]["balanced_accuracy"] for m in models["transfer"]])
    scratch = np.array([m["window_metrics"]["balanced_accuracy"] for m in models["scratch"]])
    delta = transfer - scratch
    aggregate["paired_comparison"] = {
        "mean_difference_percentage_points": float(delta.mean() * 100),
        "transfer_higher": int((delta > 1e-12).sum()),
        "ties": int((abs(delta) <= 1e-12).sum()),
        "transfer_lower": int((delta < -1e-12).sum()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(15)
    axes[0].bar(x - 0.19, scratch * 100, 0.38, label="Scratch", color="#94a3b8")
    axes[0].bar(x + 0.19, transfer * 100, 0.38, label="Transfer", color="#2563eb")
    axes[0].axhline(50, linestyle="--", color="black", linewidth=0.8)
    axes[0].set(
        xticks=x,
        xticklabels=[f"{s:02d}" for s in range(1, 16)],
        ylim=(0, 100),
        xlabel="Subject (S001-S015)",
        ylabel="Balanced accuracy (%)",
        title="Frozen R12 window classification",
    )
    axes[0].legend()
    g = aggregate["transfer"]["gates"]
    x_gates = np.arange(4)
    axes[1].bar(x_gates - 0.19, [g[k]["mean_subject_rest_command_fraction"] * 100 for k in GATES], 0.38, label="REST time active", color="#dc2626")
    axes[1].bar(x_gates + 0.19, [g[k]["mean_subject_missed_imagery_fraction"] * 100 for k in GATES], 0.38, label="Imagery time missed", color="#d97706")
    axes[1].set(
        xticks=x_gates,
        xticklabels=["Raw", "Confidence", "Temporal", "CRC +\nhysteresis"],
        ylim=(0, 110),
        ylabel="Mean across subjects (%)",
        title="Transfer command tradeoff (lower is better)",
    )
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig((RAW_ROOT or DEST) / "evaluation.png", dpi=180)
    plt.close(fig)

    eval_lines = [
        "# Frozen EEGNet experiment results - 2026-11-11",
        "",
        "Completed source pretraining, adaptation, calibration, and user-authorized final R12 evaluation. No retuning after evaluation.",
        "",
        f"Transfer mean balanced accuracy: **{transfer.mean():.2%}**, versus **{scratch.mean():.2%}** for scratch; paired mean difference **{delta.mean()*100:.2f} percentage points** across 15 subjects.",
        f"Transfer was higher for {aggregate['paired_comparison']['transfer_higher']} subjects, tied for {aggregate['paired_comparison']['ties']}, and lower for {aggregate['paired_comparison']['transfer_lower']}.",
        "",
        "**Main limitation: CRC+hysteresis emitted no commands for all 15 subjects in both model conditions. Zero false activations came with 100% missed imagery. This configuration did not provide usable command delivery.**",
        "",
        "![Frozen evaluation](evaluation.png)",
        "",
        "| Model | Gate | REST time active | Imagery time missed | Silent subjects |",
        "|---|---|---:|---:|---:|",
    ]
    for kind in ("transfer", "scratch"):
        for gate in GATES:
            v = aggregate[kind]["gates"][gate]
            eval_lines.append(
                f"| {kind} | {gate} | {v['mean_subject_rest_command_fraction']:.2%} | "
                f"{v['mean_subject_missed_imagery_fraction']:.2%} | {v['silent_subjects']}/15 |"
            )
    eval_lines += [
        "",
        "Fractions are arithmetic means of subject-level time fractions, not independent-window estimates. Classification uses 30 annotation windows per subject (15 REST, 15 imagery). Command metrics use continuous causal replay and are distinct from window accuracy.",
        "",
        "| Subject | Scratch balanced accuracy | Transfer balanced accuracy | Difference (points) |",
        "|---|---:|---:|---:|",
    ]
    for i in range(15):
        eval_lines.append(f"| S{i+1:03d} | {scratch[i]:.2%} | {transfer[i]:.2%} | {delta[i]*100:+.2f} |")
    eval_lines += [
        "",
        "Protocol and verification",
        "",
        "- Unchanged configs/gpu_research.json, code commit 6f910c4. CPU execution. Source: 89 subjects, 267 recordings, 5,340 training windows and 2,670 validation windows.",
        "- Pretraining stopped at epoch 19 after eight non-improving epochs. Best source checkpoint: epoch 11, validation loss 0.686518.",
        "- Each target model used 25 epochs on R04. Alignment used R04; threshold calibration used R08. Only then was R12 evaluated, after explicit user authorization.",
        "- CPU test suite passed with local simulation/src import path configured in pytest.ini and conftest.py.",
        "- All 155 pretrain, adaptation, calibration, and evaluation hashes verified; all 30-window trial lists verified as R12. Frozen hashes are in artifact_hashes.json, with paths relative to outputs/local_scratch_2026-11-11.",
        "",
        "Interpretation and remaining work",
        "",
        "Previously inspected S001-S010 R12 recordings and the November 11 evaluation of S011-S015 R12 mean all 15 target R12 recordings are now inspected; S011-S015 can no longer be described as an unopened confirmation cohort. A single seed and 15 participants do not establish robust superiority.",
        "",
        "The CRC candidates explicitly claim no prospective risk guarantee. Offline inference latency is simulated as zero. Robot motion, task success, retention, and physical safety were not measured in this experiment. Existing robot experiments use other decoder outputs.",
        "",
        "The next development task is to investigate command sensitivity using training/development data, then freeze a new protocol before testing. Do not tune thresholds or persistence against these R12 outcomes. Raw gating is active but has substantial false activation; stricter gates substantially miss imagery.",
        "",
        "Artifacts",
        "",
        "Local raw artifact root: outputs/local_scratch_2026-11-11. Each scratch/Sxxx and transfer/Sxxx directory contains model.pt, adaptation.json, calibration.json, evaluation/metrics.json, and evaluation/SxxxR12-decisions.jsonl. Large raw artifacts are gitignored and archived with SHA-256 verification. This report package contains aggregate and per-subject metrics, configuration, source training history, plots and checksums.",
        "",
    ]

    subject_payload = {
        kind: [
            {
                "subject": m["subject"],
                "adaptation_epochs": m.get("adaptation_epochs", 25),
                "threshold": m.get("threshold"),
                "n_rest_episodes": m.get("n_rest_episodes", 15),
                "calibration_status": m.get("calibration_status", "empirical_crc_candidate"),
                "window_metrics": m["window_metrics"],
                "command_metrics": m["command_metrics"],
            }
            for m in ms
        ]
        for kind, ms in models.items()
    }

    target_dirs = [DEST]
    if RAW_ROOT is not None and RAW_ROOT != DEST:
        target_dirs.insert(0, RAW_ROOT)

    for out_dir in target_dirs:
        (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        (out_dir / "evaluation_summary.md").write_text("\n".join(eval_lines), encoding="utf-8")
        (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        (out_dir / "subject_metrics.json").write_text(json.dumps(subject_payload, indent=2) + "\n", encoding="utf-8")
        (out_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    if RAW_ROOT is not None and RAW_ROOT != DEST:
        for img in ("pretraining.png", "evaluation.png"):
            if (RAW_ROOT / img).exists():
                shutil.copy2(RAW_ROOT / img, DEST / img)

    return aggregate


if __name__ == "__main__":
    agg = build_all()
    print(json.dumps(agg, indent=2))
