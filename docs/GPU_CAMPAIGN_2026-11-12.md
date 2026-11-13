# Large GPU training campaign

Status: launch package; no new GPU results claimed. This extends the frozen November 11 study in a separate output directory.

## Fixed matrix

- 89 source participants, 15 targets, unchanged exclusions and causal preprocessing.
- EEGNet widths F1=8/F2=16 (dropout .25) and F1=32/F2=64 (dropout .5). This is a capacity/regularization recipe comparison, not an isolated width ablation.
- Three seeds: 42, 123, 2026. Six source runs, maximum 300 epochs each, patience 60, batch 128, Adam learning rate .001.
- Target modes: scratch, full transfer, frozen-feature classifier-only transfer.
- Target learning rates .0001/.0005/.001, batch 15; score epochs 25/100/300 along the same training trajectory. With 30 windows these are 50/200/600 optimizer updates.
- 810 target trajectories; 2,430 scored target checkpoints. No early stopping on target R08: all declared milestones are reported.
- Source R04/R08 training and source R12 validation; target R04 fitting, R08 development. Target R12 is never opened by this runner. Existing inspected R12 data are not fresh confirmation data.

The dataset and networks remain compact. A large GPU enables throughput, but runtime and utilization must be measured; neither high utilization nor a long run is an outcome metric. FP32 is the initial validated path. No untested mixed-precision speedup or multi-GPU scaling claim.

## Colab

Open `notebooks/FRI_BCI_GPU_Campaign.ipynb` from the pushed branch. Select your H100 runtime, run the setup cells, authorize Drive mount, and run the campaign cell. The notebook records the exact checkout and verifies CUDA. It does not replace the runtime's PyTorch installation.

Checkpoints and logs go under `/content/drive/MyDrive/FRI_BCI/gpu_campaign_2026_11_12`. Dataset cache stays in `/content`; it can be downloaded again after a disconnect. Drive checkpoint writes favor durability over maximum speed. The same folder and same code resume the job. Colab runtime lifetimes remain limited: https://research.google.com/colaboratory/faq.html

## Brev / Linux GPU host

Use an existing CUDA-enabled PyTorch environment. Clone this repository, check out the same pinned campaign commit as Colab, and run from its root:

```bash
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
python -m eeg_decoder.gpu_campaign plan
python -u -m eeg_decoder.gpu_campaign run --work-dir /home/ubuntu/workspace/fri_campaign --cache-dir /home/ubuntu/workspace/fri_cache
python -m eeg_decoder.gpu_campaign summarize --work-dir /home/ubuntu/workspace/fri_campaign
```

Use tmux or the host's job supervisor so an SSH disconnect does not kill the process. Persistent workspace guidance: https://docs.nvidia.com/brev/latest/cli/getting-started

For multiple machines, pass `--shard N` with distinct N=0..5; each shard owns one model/seed group. Never run two writers for the same shard. Caches are separated by shard to avoid concurrent cache writes. Merge distinct shard directories under one work root before summarizing. Do not rent six H100s before timing one shard; this workload may be launch-bound. No instance is purchased or started by these scripts, and they do not stop a paid host automatically.

## Recovery and completion

Each source epoch writes optimizer/model state. Each target writes state every ten epochs and at each scored milestone. Restart the same command after interruption. A hard crash can leave `RUNNING.lock`: confirm that the prior process is stopped, then remove only that shard's lock. Normal failures release it automatically.

Code/config changes require a new work folder. Preprocessing cache validates EDF-derived artifacts; source and target checkpoints bind data hashes. Each completed shard has `complete.json` with checksums. Download or back up the work folder before deleting a host. Do not commit large checkpoints to GitHub.

Run `summarize` to produce `development_summary.json`. It marks incomplete groups; compare only groups with all 15 subjects and all three seeds. Report per-person variation and mean across seeds; seeds do not increase the independent participant count. R08 accuracy and command metrics are development estimates subject to search optimism. Keep all recipes, including failures. Accuracy alone must not select a robot gate: inspect REST activation and missed imagery together. This campaign measures fixed raw/confidence/temporal gates; CRC redesign and robot replays are separate follow-up work.

Before finalizing a winning recipe, freeze its settings and evaluation description. Any later target R12 comparison is exploratory. This package does not itself establish prospective safety, live control, or better robot performance.
