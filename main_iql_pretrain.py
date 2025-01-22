import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from tqdm import trange
from coolname import generate_slug
import wandb

import utils
from models.iql.base import IQL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")                  # Policy name
    parser.add_argument("--env", default="halfcheetah-medium-v2")   # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--save_freq", default=5e5, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=True)        # Save model and optimizer parameters
    parser.add_argument("--log_to_wb", "-w", type=bool, default=True)
    parser.add_argument("--normalize", default=False, action='store_true')
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--q_hiddens", default=2, type=int)
    parser.add_argument("--v_hiddens", default=2, type=int)
    parser.add_argument("--layernorm", default=False, action='store_true')
    parser.add_argument("--save_dir", type=str, default="./exp")
    
    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if 'antmaze' in args.env:
        replay_buffer.reward = replay_buffer.reward - 1.0
        args.q_hiddens = 3
        args.v_hiddens = 3
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1
        
    if args.log_to_wb:
        wandb_name = f"{args.env}"
        group_name = "q-pretrain"
        project_name = "QCS-Q-Pretrain"

        wandb.init(
            name=wandb_name,
            group=group_name,
            project=project_name,
            config=args
        )
        
    # Build work dir
    utils.mkdir(args.save_dir)
    base_dir = os.path.join(args.save_dir, f"{(args.policy).lower()}")
    utils.mkdir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env)
    utils.mkdir(args.work_dir)

    args.model_dir = os.path.join(args.work_dir, str(args.seed))
    utils.mkdir(args.model_dir)
        
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # IQL
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "hidden_dim": args.hidden_dim,
        "q_hiddens": args.q_hiddens,
        "v_hiddens": args.v_hiddens,
        "layernorm": args.layernorm,
    }

    # Initialize policy
    policy = IQL(**kwargs)

    for t in trange(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size, log_to_wb=args.log_to_wb)
        if (t + 1) % args.save_freq == 0:
            policy.save(args.model_dir)
    policy.save(args.model_dir)