"""Opt-in table-supported near-grasp start for learning contact and lift.

The open gripper begins near, but clear of, the free cube on the table. This
stage keeps the Approach task's observation/action/reward interface so its
checkpoint can be transferred to a farther table start after physical testing.
It is a training curriculum condition, not a table-pickup evaluation result.
"""

from isaaclab.utils import configclass

from .approach import ApproachLiftEnvCfg


# The measured scripted reference's collision-free open-gripper "grasp" pose.
# Source: outputs/isaac/eeg-matched-5p-preroll-controls-20261102T1500Z-b120s/
# scripted-positive/run.log, planned_waypoints. In that scene the fingertips
# were near z=0.3216 m and the table cube top was z=0.305 m. The same Franka
# asset/config is used here; verify actual clearance at reset before training.
NEAR_GRASP_ARM_JOINT_POS = (
    -0.07437019050121307,
    0.08722714334726334,
    -0.10742602497339249,
    -2.0976617336273193,
    0.011424562893807888,
    2.1843578815460205,
    0.5974289178848267,
)


@configclass
class NearTableLiftEnvCfg(ApproachLiftEnvCfg):
    """One fixed, table-supported near-grasp reset; no cube attachment."""

    def __post_init__(self):
        super().__post_init__()
        self._near_grasp_robot_start()

    def _near_grasp_robot_start(self):
        joint_pos = self.scene.robot.init_state.joint_pos
        for i, value in enumerate(NEAR_GRASP_ARM_JOINT_POS, start=1):
            joint_pos[f"panda_joint{i}"] = value
        joint_pos["panda_finger_joint.*"] = 0.04

    def play_mode(self):
        super().play_mode()
        self._near_grasp_robot_start()
