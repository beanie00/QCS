import torch
import random
import numpy as np
import torch
import gym

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_env_spec(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()) + 1e-6,
        float(env.action_space.high.max()) - 1e-6,
    ]
    env.close()
    return state_dim, act_dim, action_range


def get_env_builder(seed, env_name, target_goal=None):
    def make_env_fn():
        env = gym.make(env_name)
        env.seed(seed)
        if hasattr(env.env, "wrapped_env"):
            env.env.wrapped_env.seed(seed)
        elif hasattr(env.env, "seed"):
            env.env.seed(seed)
        else:
            pass
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        if target_goal:
            env.set_target_goal(target_goal)
            print(f"Set the target goal to be {env.target_goal}")
        return env

    return make_env_fn