"""Fixed table-cube lift task for a bounded physical-control pilot."""

from isaaclab.utils import clone, configclass
from isaaclab_tasks.core.lift import lift_env_cfg as lift
from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FrankaLiftEnvCfg


@configclass
class TableLiftEnvCfg(FrankaLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._fixed_table_cube()

    def _fixed_table_cube(self):
        # Table top 0.255 m; center of a resting 0.05 m cube 0.28 m.
        self.scene.object.spawn = clone(lift.ObjectCfg().cube)
        self.scene.object.init_state.pos = (-0.55, 0.10, 0.28)
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (0.24, 2.0)

        # Commands are in the Franka base frame, rotated pi about world z.
        target = self.commands.object_pose.ranges
        target.pos_x = (0.55, 0.55)
        target.pos_y = (-0.10, -0.10)
        target.pos_z = (0.43, 0.43)

        reset = self.events.conditional_reset.params
        terms = reset["terms"]
        terms.pop("reset_object_to_target", None)
        terms["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        terms["reset_robot_wrist_joint"].params["position_range"] = (0.0, 0.0)
        terms["reset_gripper_width"].params["position_range"] = (0.0, 0.0)
        terms["reset_object"].params["pose_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        reset["buffer_size_per_group"] = 16
        reset["oversample_factor"] = 1.0
        reset["diversity_feature"] = None
        # The upstream 2 cm slab margin rejects an object touching the table.
        reset["valid_criteria"]["robot_table_clearance"].min_clearance = -0.003
        reset["max_prefill_iters"] = 128

        self.events.variable_gravity.params["gravity_distribution_params"] = (
            (0.0, 0.0, -9.81), (0.0, 0.0, -9.81)
        )
        for name in (
            "robot_physics_material", "object_physics_material", "object_physics_inertia",
            "joint_stiffness_and_damping", "joint_friction", "object_scale_mass",
            "gripper_closing_speed",
        ):
            setattr(self.events, name, None)
        self.curriculum = None

    def play_mode(self):
        super().play_mode()
        self._fixed_table_cube()
