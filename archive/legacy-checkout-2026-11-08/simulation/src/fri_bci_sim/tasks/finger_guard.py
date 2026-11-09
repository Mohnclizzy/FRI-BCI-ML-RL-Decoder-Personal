"""Playback diagnostic that bounds finger targets for a fixed 5-cm cube.

The actor still receives the same observations and emits the same nine relative
joint actions. Only the two final absolute finger position targets are bounded.
This variant is for a measured inference intervention; no policy was trained
under its dynamics. It must remain separate from the progress-reward task.
"""

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass

from .near_table_progress import NearTableLiftProgressEnvCfg


FINGER_TARGET_MIN_M = 0.02
FINGER_TARGET_MAX_M = 0.04


class FingerGuardRelativeJointPositionAction(RelativeJointPositionAction):
    """Clip absolute finger targets after the usual relative-action mapping."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = list(self._joint_names)
        self._finger_columns = tuple(names.index(name) for name in
                                     ("panda_finger_joint1", "panda_finger_joint2"))
        if len(names) != 9 or self._finger_columns != (7, 8):
            raise RuntimeError(f"Unexpected Franka action order: {names}")

    def apply_actions(self):
        targets = self.processed_actions + self._asset.data.joint_pos.torch[:, self._joint_ids]
        targets[:, self._finger_columns] = targets[:, self._finger_columns].clamp(
            min=FINGER_TARGET_MIN_M, max=FINGER_TARGET_MAX_M
        )
        self._asset.set_joint_position_target_index(target=targets, joint_ids=self._joint_ids)


@configclass
class NearTableFingerGuardEnvCfg(NearTableLiftProgressEnvCfg):
    """Progress-reward task with only the finger target mapping altered."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.action.class_type = FingerGuardRelativeJointPositionAction

    def play_mode(self):
        super().play_mode()
        self.actions.action.class_type = FingerGuardRelativeJointPositionAction
