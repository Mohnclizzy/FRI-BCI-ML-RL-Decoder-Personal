"""A deliberately simple PyBullet proxy gripper for grasp/hold experiments.

The gripper is kinematic and a close action creates a fixed grasp constraint
when it encloses the cube. This is a task prototype, not a dexterous-hand or
contact-rich Isaac Lab result.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pybullet as bullet
from gymnasium import spaces

from rl_policy import HAND_IMAGERY, REST


class GraspHoldEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, error_rest: float = 0.0, error_imagery: float = 0.0, render_mode=None):
        super().__init__()
        if not (0 <= error_rest <= 1 and 0 <= error_imagery <= 1):
            raise ValueError("Error probabilities must be between 0 and 1")
        self.error_rest = error_rest
        self.error_imagery = error_imagery
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)  # hold, close, lift
        # command, gripper height, closed, attached, cube height, time remaining
        self.observation_space = spaces.Box(0, 1, (6,), dtype=np.float32)
        self.client = bullet.connect(bullet.DIRECT)
        bullet.setGravity(0, 0, -9.81, physicsClientId=self.client)
        bullet.setTimeStep(1 / 120, physicsClientId=self.client)
        self.step_count = 0
        self.true_intent = REST
        self.command = REST
        self.cube = None
        self.gripper = None
        self.jaws = []
        self.grasp_constraint = None

    def _box(self, half_extents, position, mass=0, color=(0.5, 0.5, 0.5, 1)):
        shape = bullet.createCollisionShape(bullet.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client)
        visual = bullet.createVisualShape(bullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client)
        return bullet.createMultiBody(mass, shape, visual, position, physicsClientId=self.client)

    def _place_gripper(self):
        bullet.resetBasePositionAndOrientation(self.gripper, [0, 0, self.height], [0, 0, 0, 1], physicsClientId=self.client)
        gap = 0.043 if self.closed else 0.12
        for sign, jaw in zip((-1, 1), self.jaws):
            bullet.resetBasePositionAndOrientation(jaw, [sign * gap / 2, 0, self.height - 0.035], [0, 0, 0, 1], physicsClientId=self.client)

    def _observation(self):
        cube_height = bullet.getBasePositionAndOrientation(self.cube, physicsClientId=self.client)[0][2]
        return np.array([
            self.command, self.height / 0.30, float(self.closed),
            float(self.grasp_constraint is not None), cube_height / 0.30,
            (8 - self.step_count) / 8,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.step_count = 0
        self.height = 0.10
        self.closed = False
        self.grasp_constraint = None
        self.true_intent = int(options.get("true_intent", self.np_random.integers(0, 2)))
        if self.true_intent not in (REST, HAND_IMAGERY):
            raise ValueError("Unknown true intent")
        error = self.error_rest if self.true_intent == REST else self.error_imagery
        self.command = int(options.get("command", self.true_intent ^ int(self.np_random.random() < error)))
        if self.command not in (REST, HAND_IMAGERY):
            raise ValueError("Unknown decoded command")

        bullet.resetSimulation(physicsClientId=self.client)
        bullet.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self._box([0.3, 0.3, 0.01], [0, 0, -0.01])
        self.cube = self._box([0.025, 0.025, 0.025], [0, 0, 0.026], 0.05, (0.8, 0.2, 0.2, 1))
        self.gripper = self._box([0.015, 0.04, 0.008], [0, 0, self.height], 0, (0.2, 0.2, 0.8, 1))
        self.jaws = [self._box([0.006, 0.025, 0.035], [0, 0, 0], 0, (0.2, 0.2, 0.8, 1)) for _ in range(2)]
        # Jaw contact is represented by the explicit enclosure condition below.
        for jaw in self.jaws:
            bullet.setCollisionFilterPair(jaw, self.cube, -1, -1, 0, physicsClientId=self.client)
        self._place_gripper()
        for _ in range(12):
            bullet.stepSimulation(physicsClientId=self.client)
        return self._observation(), {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Unknown action")
        self.step_count += 1
        acquired = False
        if action == 1 and not self.closed:
            self.closed = True
            cube_z = bullet.getBasePositionAndOrientation(self.cube, physicsClientId=self.client)[0][2]
            if abs(self.height - 0.10) < 0.015 and cube_z < 0.05:
                self.grasp_constraint = bullet.createConstraint(
                    self.gripper, -1, self.cube, -1, bullet.JOINT_FIXED,
                    [0, 0, 0], [0, 0, -0.074], [0, 0, 0], physicsClientId=self.client,
                )
                acquired = True
        elif action == 2 and self.closed:
            self.height = min(self.height + 0.035, 0.30)
        self._place_gripper()
        for _ in range(12):
            bullet.stepSimulation(physicsClientId=self.client)
        cube_z = bullet.getBasePositionAndOrientation(self.cube, physicsClientId=self.client)[0][2]
        lifted = cube_z > 0.16
        disturbed = self.closed or self.height > 0.11 or cube_z > 0.06
        terminated = lifted or self.step_count >= 8
        if self.true_intent == HAND_IMAGERY:
            reward = 0.1 * acquired + (0.04 if self.grasp_constraint is not None and action == 2 else 0)
            reward += 1.0 if lifted else 0.0
        else:
            reward = -0.25 if disturbed and action != 0 else 0.0
            reward += 1.0 if terminated and not disturbed else 0.0
        reward -= 0.005
        info = {"success": bool(lifted if self.true_intent == HAND_IMAGERY else not disturbed),
                "lifted": bool(lifted), "disturbed_rest": bool(disturbed),
                "true_intent": self.true_intent, "command": self.command}
        return self._observation(), float(reward), terminated, False, info

    def render(self):
        _, _, rgba, _, _ = bullet.getCameraImage(
            320, 240, bullet.computeViewMatrixFromYawPitchRoll([0, 0, 0.1], 0.5, 45, -30, 0, 2),
            bullet.computeProjectionMatrixFOV(60, 320 / 240, 0.01, 2),
            renderer=bullet.ER_TINY_RENDERER, physicsClientId=self.client,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(240, 320, 4)[:, :, :3]

    def close(self):
        if bullet.isConnected(self.client):
            bullet.disconnect(self.client)
