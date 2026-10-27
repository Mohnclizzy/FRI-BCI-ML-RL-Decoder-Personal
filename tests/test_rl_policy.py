from rl_policy.env import GraspHoldEnv
from rl_policy.evaluate import measure, run_episode


def test_balanced_scripted_controls():
    env = GraspHoldEnv()
    trials = [("rest", 0, 0), ("imagery", 1, 1)]
    try:
        assert measure(env, trials, "always_hold")["balanced_success"] == 0.5
        assert measure(env, trials, "always_grasp")["balanced_success"] == 0.5
        result = run_episode(env, "always_grasp", 1, 1)
        assert result["lifted"] and result["success"]
    finally:
        env.close()


def test_observation_contains_command_but_not_hidden_intent():
    env = GraspHoldEnv()
    try:
        rest_obs, _ = env.reset(seed=5, options={"true_intent": 0, "command": 0})
        imagery_obs, _ = env.reset(seed=5, options={"true_intent": 1, "command": 0})
        assert (rest_obs == imagery_obs).all()
    finally:
        env.close()
