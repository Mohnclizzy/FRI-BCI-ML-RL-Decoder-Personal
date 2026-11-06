"""Near-table PPO diagnostic penalizing lateral cube displacement."""
import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from .near_table_progress import NearTableLiftProgressEnvCfg


def cube_horizontal_displacement(env):
    """Bounded distance from the fixed table reset in the environment frame."""
    pos = env.scene["object"].data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    origin = pos.new_tensor((-0.55, 0.10))
    return torch.tanh(torch.linalg.vector_norm(pos - origin, dim=-1) / 0.05)


@configclass
class NearTableAntiPushEnvCfg(NearTableLiftProgressEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.cube_horizontal_displacement = RewardTermCfg(
            func=cube_horizontal_displacement, weight=-2.0
        )
