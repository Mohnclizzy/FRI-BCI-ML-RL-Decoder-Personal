# Repository layout and artifact recovery

The source of record is [FRI-BCI-ML-RL-Decoder-Personal](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal), branch `codex/robot-learning-design`. The repository is public as checked on November 8, 2026. Do not add credentials or private account exports.

## Where files belong

| Location | Contents |
| --- | --- |
| `README.md`, `START_HERE.md` | Navigation and setup |
| `STATUS.md` | Dated execution status; later entries supersede earlier plans |
| `eeg_decoder/`, `simulation/` | Current executable code and simulator integration |
| `configs/`, `requirements*.txt` | Configuration and reproducible dependencies |
| `notebooks/` | Colab and interactive workflows |
| `analysis/`, `scripts/`, `tests/` | Analysis, maintenance, and verification |
| `report/midterm/` | Current submitted PDF, TeX sources, bibliography, styles, and Overleaf ZIP |
| `docs/` | Protocols, handoffs, literature, and operational documentation |
| `results/` | Versioned summaries, metrics, plots, and supporting evidence |
| `artifacts/` | Small curated checkpoints and artifact manifests |
| `outputs/isaac/` | Explicitly allowlisted decision streams; other generated output stays ignored |
| `archive/` | Historical local versions and handoffs, clearly separated from current code |
| `continuity/` | Previously committed recovery snapshots, retained at their existing paths |
| GitHub Releases | Bulk raw outputs, downloaded public data, historical snapshots, and recovery bundles |

Existing runtime paths are preserved so notebooks and replay scripts continue to resolve their inputs. Historical copies in `archive/` must not be used as the current implementation. The root `main.tex` remains a historical document; the current midterm source is `report/midterm/midterm_report.tex`.

## November 8 local consolidation

[Download the release](https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal/releases/tag/local-consolidation-2026-11-08).

| Asset | Contents | Restore destination |
| --- | --- | --- |
| `experiment-outputs.zip` | Locally available raw experiment output, policies, telemetry, and decoder artifacts | A separate historical checkout; entries begin with `outputs/` |
| `physionet-data.zip` | The 15 locally cached public PhysioNet recordings | Checkout root; entries begin with `data/` |
| `legacy-source.zip` | Original older checkout source and untracked project files | A separate historical directory |
| `workspace-handoffs.zip` | Loose project handoffs, audit material, screenshots, and prior archive | A separate historical directory |
| `legacy-repository.bundle` | All saved refs and history from the older checkout | `git clone legacy-repository.bundle recovered-history` |
| `SHA256SUMS.txt` | Archive checksums | Verify before extraction |
| `local-consolidation-2026-11-08.json` | Per-file source paths, sizes, hashes, archive members, and exclusions | Audit / recovery index |

The same manifest is versioned in [`artifacts/manifests/`](../artifacts/manifests/local-consolidation-2026-11-08.json). Archive entries were read back and hashed after compression. The files cover the known local project workspace, not all of AntiHermes or every folder on the computer. They do not establish the previously claimed 198-checkpoint inventory.

Virtual environments, Git internals, Python caches, notebook checkpoints, and installed Node dependencies are excluded from the file archives. The legacy Git bundle preserves repository history without publishing local Git configuration or credential helpers. Recreate environments from the dependency files.

## Restore only what you need

```sh
git clone --branch codex/robot-learning-design https://github.com/Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal.git
gh release download local-consolidation-2026-11-08 --repo Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal --pattern SHA256SUMS.txt --pattern experiment-outputs.zip
```

Compare the downloaded archive's SHA-256 with `SHA256SUMS.txt` using `sha256sum` or PowerShell `Get-FileHash -Algorithm SHA256`. Extract historical output into a separate directory first to avoid overwriting newer streams. Ordinary source clones do not download release assets.

Future experiment batches should publish compressed raw artifacts as release assets and commit their checksums and summaries. Do not add whole environments or repeatedly copy the same experiment into multiple source directories.
