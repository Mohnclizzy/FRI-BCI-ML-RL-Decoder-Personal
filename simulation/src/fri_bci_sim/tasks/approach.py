"""Table-lift PPO variant with a stronger approach-and-contact learning signal.

This task deliberately keeps the Franka, observations, actions, resets and
checkpoint input dimensions identical to ``FRI-Franka-Lift``. Only the reward
weights change, so a measured pre-grasp checkpoint can still initialize table
fine-tuning while the original task remains available as a baseline.
"""

from isaaclab.utils import configclass

from .lift import TableLiftEnvCfg


@configclass
class ApproachLiftEnvCfg(TableLiftEnvCfg):
    """Table task that gives approach/contact behavior a usable PPO signal."""

    def __post_init__(self):
        super().__post_init__()
        # In the inherited task, fingertip-to-object shaping has weight 0.05,
        # while object-to-goal progress has weight 5.0. On table starts the
        # cube cannot make goal progress until the arm reaches and grasps it.
        # Strengthen the dense reach and bilateral-contact terms to bridge that
        # otherwise weakly supervised phase. Keep object goal-progress and
        # terminal-success terms unchanged for the post-grasp lift.
        self.rewards.fingers_to_object.weight = 1.0
        self.rewards.good_finger_contact.weight = 2.0
        self.rewards.contact_count.weight = 0.25

    def play_mode(self):
        super().play_mode()
