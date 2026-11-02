"""Opt-in physical pre-grasp stage for learning to close and lift first.

The cube is only *placed* between the fingers at reset.  It remains a free
rigid body with contacts and full gravity; no constraint attaches it to the hand.
"""

from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import clone, configclass
from isaaclab_tasks.core.lift import mdp

from .lift import TableLiftEnvCfg


@configclass
class PregraspLiftEnvCfg(TableLiftEnvCfg):
    """Same single-cube Franka task as table pickup, with in-hand starts.

    The observation groups, relative joint actions, reward terms and PPO agent
    configuration are inherited unchanged, so its checkpoint can initialize the
    table-only stage.  These starts are privileged training starts and must not
    be counted as table-pickup success.
    """

    def __post_init__(self):
        super().__post_init__()
        self._use_pregrasp_starts()

    def _use_pregrasp_starts(self):
        terms = self.events.conditional_reset.params["terms"]
        # The upstream reset_to_grasp must follow the arm and width resets: it
        # reads the current hand pose, then writes the coupled finger pair and
        # the free object's root pose/velocity for selected environments.
        terms["reset_object_to_target"] = EventTerm(
            func=mdp.reset_to_grasp,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "gripper_cfg": SceneEntityCfg("robot", joint_names="panda_finger_joint.*"),
                "probability": 1.0,
                "pose_range": {"x": (-0.002, 0.002), "y": (-0.002, 0.002), "z": (0.1, 0.1)},
                "grasp_configs": [(clone(self.scene.object.spawn), 0.026, (0.0, 0.0, 0.0, 1.0))],
            },
        )
        # A high goal makes the stage explicitly about lifting from the hand
        # rather than reaching the table goal at world z=0.43 m.
        self.commands.object_pose.ranges.pos_z = (0.75, 0.75)

    def play_mode(self):
        super().play_mode()
        # TableLiftEnvCfg.play_mode re-applies its table-only reset; restore the
        # pre-grasp stage for explicit pre-grasp playback as well.
        self._use_pregrasp_starts()
