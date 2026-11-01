"""Physical lift baseline inheriting the pinned upstream scene, MDP and PPO config.

No fixed grasp attachment is created by this project. The pinned upstream task
uses simulated contact and actuator dynamics. Its default mixed pre-grasp reset
would confound claims of picking an object from a table, so it is disabled here.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FrankaLiftEnvCfg


@configclass
class TableLiftEnvCfg(FrankaLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._table_only()

    def _table_only(self):
        reset = self.events.conditional_reset.params["terms"]["reset_object_to_target"]
        reset.params["probability"] = 0.0

    def play_mode(self):
        super().play_mode()
        self._table_only()
