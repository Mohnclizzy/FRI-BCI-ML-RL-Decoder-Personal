# Pre-grasp RL pilot — 2026-11-02

Ran on the existing Brev L4 with Isaac Lab `v3.0.0` commit
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Newton backend, project commit
`8b9db42` copied as a Git archive to `/home/ubuntu/FRI-project-8b9db42`.

Command: `python -m fri_bci_sim.curriculum --isaaclab-root
/home/ubuntu/IsaacLab-fri --stage pregrasp --num-envs 64 --seed 8
--iterations 100 --run-dir outputs/isaac/pregrasp-64x100-seed8 --execute`.
The process exited successfully and saved `model_0.pt` and `model_99.pt`.

This is only a short PPO engineering pilot on **pre-grasp starts** where the
free cube is initialized between the fingers. It is not a table pickup test.
Training-reported success peaked near 0.115 then declined to 0.092 by the last
reported iteration. `object_out_of_bound` terminations rose to about 0.39 and
object position error remained around 0.58. These are training diagnostics,
not independent physical evaluation. No policy playback or table-stage transfer
was run. Do not claim the policy learned a reliable lift or promote this
checkpoint without evaluation.

Outputs were copied locally under ignored
`outputs/isaac/pregrasp-64x100-seed8/` and remain on the stopped Brev instance
under the corresponding project output path. Brev stop was requested after the
run; check `brev ls` for STOPPED before assuming compute billing ended.

Next: inspect reset geometry and out-of-bounds episodes, make a short measured
pre-grasp checkpoint evaluation, then test the table stage only if it achieves
repeated sustained free-object lifts with bilateral contact. Keep each stage's
success counts separate.
