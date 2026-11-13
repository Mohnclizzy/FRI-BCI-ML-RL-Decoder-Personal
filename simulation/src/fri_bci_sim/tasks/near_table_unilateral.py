"""Near-table anti-push task with a unilateral-contact penalty."""
import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from .near_table_antipush import NearTableAntiPushEnvCfg


def unilateral_contact(env):
    """One when exactly one filtered finger sensor exceeds 0.1 N."""
    forces = []
    for name in ('panda_leftfinger_object_s', 'panda_rightfinger_object_s'):
        data = env.scene.sensors[name].data.normal_force_matrix_w
        forces.append(torch.linalg.vector_norm(data.torch.reshape(env.num_envs, -1), dim=1) >= 0.1)
    return torch.logical_xor(*forces).float()


@configclass
class NearTableUnilateralEnvCfg(NearTableAntiPushEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.unilateral_contact = RewardTermCfg(func=unilateral_contact, weight=-1.0)
