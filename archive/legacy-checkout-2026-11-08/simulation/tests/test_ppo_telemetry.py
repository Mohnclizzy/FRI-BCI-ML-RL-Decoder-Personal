import unittest

from fri_bci_sim.ppo_telemetry import EpisodeAccumulator, _split_joint_targets


class EpisodeAccumulatorTests(unittest.TestCase):
    def _qualifying_samples(self, accumulator, count, *, start=0, z=0.36, contacts=(0.12, 0.11)):
        for index in range(start, start + count):
            accumulator.add_sample((index + 1) * 0.005, z, list(contacts))

    def test_reset_height_is_baseline_and_lift_uses_elapsed_time(self):
        accumulator = EpisodeAccumulator(episode=1, baseline_z=0.28, physics_dt=0.005)
        self._qualifying_samples(accumulator, 100)

        result = accumulator.finish(status="completed")

        self.assertTrue(result["physical_lift_pass"])
        self.assertAlmostEqual(result["baseline_cube_height_m"], 0.28)
        self.assertAlmostEqual(result["acquisition_time_s"], 0.5)
        self.assertAlmostEqual(result["best_sustained_lift_contact_s"], 0.5)
        self.assertEqual(result["contact_coverage"], 1.0)

    def test_physics_gap_breaks_sustained_contact(self):
        accumulator = EpisodeAccumulator(episode=1, baseline_z=0.28, physics_dt=0.005)
        accumulator.add_sample(0.005, 0.36, [0.12, 0.11])
        accumulator.add_sample(0.025, 0.36, [0.12, 0.11])

        result = accumulator.finish(status="incomplete", reason="early_stop")

        self.assertFalse(result["physical_lift_pass"])
        self.assertEqual(result["physics_step_gaps"], 1)
        self.assertAlmostEqual(result["best_sustained_lift_contact_s"], 0.005)

    def test_unrounded_physics_clock_does_not_accumulate_rounding_error(self):
        accumulator = EpisodeAccumulator(episode=1, baseline_z=0.28, physics_dt=1 / 120)
        for index in range(1440):
            accumulator.add_sample((index + 1) / 120, 0.28, [0.0, 0.0])

        result = accumulator.finish(status="completed")

        self.assertAlmostEqual(result["duration_s"], 12.0, places=9)

    def test_drop_requires_elapsed_time_after_acquisition(self):
        accumulator = EpisodeAccumulator(episode=1, baseline_z=0.28, physics_dt=0.005)
        self._qualifying_samples(accumulator, 100)
        for index in range(100, 150):
            accumulator.add_sample((index + 1) * 0.005, 0.30, [0.0, 0.0])

        result = accumulator.finish(status="completed")

        self.assertEqual(result["drop_count"], 1)
        self.assertAlmostEqual(result["drop_times_s"][0], 0.75)

    def test_contact_coverage_and_incomplete_reason_are_explicit(self):
        accumulator = EpisodeAccumulator(episode=2, baseline_z=0.28, physics_dt=0.005)
        accumulator.add_sample(0.005, 0.28, [None, None])
        accumulator.add_sample(0.010, 0.28, [0.0, 0.0])

        result = accumulator.finish(status="incomplete", reason="max_steps_exhausted")

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "max_steps_exhausted")
        self.assertEqual(result["contact_missing_samples"], 1)
        self.assertEqual(result["contact_coverage"], 0.5)

    def test_playback_error_is_retained_in_incomplete_summary(self):
        accumulator = EpisodeAccumulator(episode=3, baseline_z=0.28, physics_dt=0.005)
        result = accumulator.finish(
            status="incomplete",
            reason="step_exception",
            error="RuntimeError: sensor read failed",
            policy_steps=7,
        )

        self.assertEqual(result["reason"], "step_exception")
        self.assertEqual(result["error"], "RuntimeError: sensor read failed")
        self.assertEqual(result["policy_steps"], 7)

    def test_joint_targets_are_split_into_arm_and_fingers(self):
        names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        targets, arm, fingers = _split_joint_targets(names, [float(i) for i in range(9)])

        self.assertEqual(len(targets), 9)
        self.assertEqual(len(arm), 7)
        self.assertEqual(len(fingers), 2)
        self.assertEqual(fingers["panda_finger_joint2"], 8.0)


if __name__ == "__main__":
    unittest.main()
