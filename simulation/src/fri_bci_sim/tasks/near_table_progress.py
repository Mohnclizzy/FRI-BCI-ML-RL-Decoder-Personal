"""Near-table lift experiment with progress favored over holding contact.

This is a separate task ID so the original near-table run's rewards and
checkpoint retain their exact meaning. Robot, cube, reset, observations,
actions, physics, and success criterion are inherited unchanged.
"""

from isaaclab.utils import configclass

from .near_table import NearTableLiftEnvCfg


@configclass
class NearTableLiftProgressEnvCfg(NearTableLiftEnvCfg):
    """Reduce stationary contact payment and increase credited goal progress."""

    def __post_init__(self):
        super().__post_init__()
        self._progress_rewards()

    def _progress_rewards(self):
        # The first 2.048M-transition near-table run earned far more from
        # continuous contact than from cube-to-goal progress, and five complete
        # physical screens showed contact without lifting. Keep reach shaping
        # but lower the stationary terms and increase the existing episodic
        # best-progress term, which requires bilateral finger contact.
        self.rewards.fingers_to_object.weight = 0.2
        self.rewards.good_finger_contact.weight = 0.1
        self.rewards.contact_count.weight = 0.0
        self.rewards.position_tracking.weight = 25.0

    def play_mode(self):
        super().play_mode()
        self._progress_rewards()
