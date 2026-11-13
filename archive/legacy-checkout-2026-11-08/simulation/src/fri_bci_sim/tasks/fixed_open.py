"""Opt-in playback intervention that holds both Franka fingers open at 0.04 m.

The policy still emits the unchanged nine joint actions. The seven arm values
retain RelativeJointPositionAction semantics; the two finger values are ignored
for target generation and replaced by the fixed 0.04 m absolute targets.
"""

import torch
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass

from .near_table_progress import NearTableLiftProgressEnvCfg


FIXED_FINGER_TARGET_M = 0.04
FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")


class FixedOpenRelativeJointPositionAction(RelativeJointPositionAction):
    """Preserve all seven arm commands and set both finger targets to 0.04 m."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = list(self._joint_names)
        if len(names) != 9 or tuple(names[-2:]) != FINGER_JOINT_NAMES:
            raise RuntimeError(f"Unexpected nine-action Franka joint order: {names}")
        self._finger_columns = (7, 8)

    def apply_actions(self):
        arm_targets = (
            self.processed_actions[:, :7]
            + self._asset.data.joint_pos.torch[:, self._joint_ids][:, :7]
        )
        targets = torch.empty_like(self.processed_actions)
        targets[:, :7] = arm_targets
        targets[:, self._finger_columns] = FIXED_FINGER_TARGET_M
        self._asset.set_joint_position_target_index(target=targets, joint_ids=self._joint_ids)


@configclass
class NearTableFixedOpenEnvCfg(NearTableLiftProgressEnvCfg):
    """Progress task with fixed-open playback action mapping only."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.action.class_type = FixedOpenRelativeJointPositionAction

    def play_mode(self):
        super().play_mode()
        self.actions.action.class_type = FixedOpenRelativeJointPositionAction
