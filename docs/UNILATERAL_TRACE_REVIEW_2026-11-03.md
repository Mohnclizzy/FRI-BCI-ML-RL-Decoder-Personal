# Matched-start trace review — 2026-11-03

The claimed 2/5 anti-push versus 0/5 Approach comparison is **not paired**. The Approach run used one process at seed 8 for five sequential resets with the farther ready-arm pose; anti-push used five processes at seeds 8–12 from a near-grasp arm pose. Both started the cube at (-0.55, 0.10, 0.28) m and fingers at 0.04 m. The recorded Approach arm joints start [0, -0.569, 0, -2.810, 0, 3.037, 0.741], while near-table starts [-0.0744, 0.0872, -0.1074, -2.0977, 0.0114, 2.1844, 0.5974] radians. Those are physically different tasks.

The Progress near-table screen is the more comparable control: seeds 8–12 and all recorded initial cube poses and joint positions match corresponding anti-push episodes exactly. Its physical outcomes were also 2/5 sustained acquisitions, one with a subsequent drop. This matches logged reset fields and seed labels; it does not establish matched simulator RNG streams, sensor histories, solver internals, or prove a causal effect. Independent policy and reward training histories differ.

The table below reports first sampled times in seconds: R = right >=0.1 N with left <0.1 N; P = cube XY displacement >=0.02 m while rise <0.04 m; C = either actual finger joint <=0.020 m (a descriptive closure marker, not proof of grasp); L = first rise >=0.08 m; A = sustained bilateral acquisition; D = post-acquisition drop detection. Dashes mean the sampled event was absent. Contact is filtered sensor magnitude; thresholds are sampled at 120 Hz. Events can overlap or recur; first occurrence does not imply their subsequent continuity. The JSON record includes finger positions and contact forces at each event, both-finger closure timing, reset poses and trace hashes.

| Condition | Seed / episode | R | P | C | L | A | D |
|---|---|---:|---:|---:|---:|---:|---:|
| Approach | 8 / 1 | 0.775 | 1.075 | 1.275 | — | — | — |
| Approach | 8 / 2 | 0.683 | 0.958 | 0.983 | — | — | — |
| Approach | 8 / 3 | 0.925 | 1.275 | 1.092 | — | — | — |
| Approach | 8 / 4 | 0.558 | 0.775 | 1.100 | — | — | — |
| Approach | 8 / 5 | 0.583 | 0.842 | 1.742 | — | — | — |
| Progress | 8 | 0.050 | 0.108 | 0.775 | — | — | — |
| Progress | 9 | 0.083 | 0.167 | 1.342 | — | — | — |
| Progress | 10 | 0.075 | 0.175 | 2.208 | 1.283 | 1.775 | 2.500 |
| Progress | 11 | 0.075 | 0.158 | — | 1.033 | 1.533 | — |
| Progress | 12 | 0.100 | 0.183 | 1.242 | — | — | — |
| Anti-push | 8 | 0.050 | 0.117 | 0.258 | 0.392 | 1.000 | — |
| Anti-push | 9 | 0.117 | 0.192 | 1.092 | — | — | — |
| Anti-push | 10 | 0.117 | 0.183 | — | — | — | — |
| Anti-push | 11 | 0.142 | 0.200 | — | — | — | — |
| Anti-push | 12 | 0.150 | 0.200 | 2.742 | 0.958 | 1.450 | 3.017 |

The early unilateral contact and displacement precede the closure marker in the near-table failures, and often precede lift even in the two acquired trajectories. This supports testing a unilateral-contact penalty, not a claim that finger closure alone caused the push. Sensor contact and 20 mm joint positions do not identify actual contact surfaces or grasp stability. The fixed-open and two-step arm filter diagnostics also showed that simple early finger or arm clamps are insufficient. No physical safety or generalization conclusion follows from these small fixed-pose screens.

Reproduce with `python analysis/compare_antipush_events.py`; results are in [trace comparison JSON](../results/trace_comparison_2026-11-03.json), and raw inputs remain in the existing ignored output directories.
