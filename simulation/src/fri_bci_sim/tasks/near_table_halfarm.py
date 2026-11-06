"""Inference diagnostic: halve the arm's relative position increments."""
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass
from .near_table_unilateral import NearTableUnilateralEnvCfg


def half_arm_increments(processed):
    if processed.ndim != 2 or processed.shape[1] != 9:
        raise ValueError('expected nine Franka action components')
    result = processed.clone()
    result[:, :7] *= 0.5
    return result


class HalfArmRelativeAction(RelativeJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if list(self._joint_names) != [*(f'panda_joint{i}' for i in range(1,8)), 'panda_finger_joint1', 'panda_finger_joint2']:
            raise ValueError(f'unexpected action order: {self._joint_names}')

    def apply_actions(self):
        target = half_arm_increments(self.processed_actions) + self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class NearTableHalfArmEnvCfg(NearTableUnilateralEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.action.class_type = HalfArmRelativeAction

    def play_mode(self):
        super().play_mode()
        self.actions.action.class_type = HalfArmRelativeAction
