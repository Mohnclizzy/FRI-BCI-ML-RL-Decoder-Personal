"""Discovered by Isaac Lab's external task entry-point mechanism."""

import gymnasium as gym

gym.register(
    id="FRI-Franka-Lift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.lift:TableLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-Pregrasp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.pregrasp:PregraspLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-Approach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.approach:ApproachLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-NearTable",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.near_table:NearTableLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-NearTable-Progress",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.near_table_progress:NearTableLiftProgressEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-NearTable-Progress-FingerGuard",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.finger_guard:NearTableFingerGuardEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-NearTable-AntiPush",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.near_table_antipush:NearTableAntiPushEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)


gym.register(
    id="FRI-Franka-Lift-NearTable-Unilateral",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.near_table_unilateral:NearTableUnilateralEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="FRI-Franka-Lift-NearTable-HalfArm",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "fri_bci_sim.tasks.near_table_halfarm:NearTableHalfArmEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.lift.config.franka.agents.rsl_rl_ppo_cfg:FrankaPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
