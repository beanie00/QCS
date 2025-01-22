"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import argparse
import time
import gym
import numpy as np
import wandb

from stable_baselines3.common.vec_env import SubprocVecEnv
from trainer.main_trainer import Maintrainer
import utils
from utils import (
    Logger,
    create_model,
    create_optimizer,
    create_scheduler,
    get_env_spec,
    get_env_builder,
    load_dataset,
    initialize_q_network,
    get_q_loss_mean
)

MAX_EPISODE_LEN = 1000


class Experiment:
    def __init__(self, variant):
        self.variant = variant
        self.start_time = time.time()
        self._initialize_experiment_parameters()
        self._initialize_environment()
        self._initialize_model()
        self._initialize_dataset()
        self._initialize_training_components()
        self._initialize_logging()

    def _initialize_experiment_parameters(self):
        self.iter = 0
        # Set reward scale based on the environment
        env = self.variant.get("env", "")
        if "antmaze" in env:
            self.reward_scale = 1.0
        elif "kitchen" in env:
            self.reward_scale = 0.01
        else:
            self.reward_scale = 0.001

        # Set architecture-specific parameters
        base_arch = self.variant.get('base_arch', "")
        if 'mlp' in base_arch:
            context_length = 1
        elif 'dc' in base_arch:
            context_length = 8
        else:
            context_length = None  # Default or error handling as needed

        if context_length is not None:
            self.variant.update({
                'K': context_length,
                'eval_context_length': context_length
            })

        # Set environment-specific configurations
        if "antmaze" in env:
            batch_size, lr, max_iters = 256, 0.0003, 1000
        elif "pen" in env or "kitchen" in env:
            batch_size, lr, max_iters = 64, 0.0003, 500
        else:
            batch_size, lr, max_iters = 64, 0.0001, 500

        self.variant.update({
            'batch_size': batch_size,
            'learning_rate': lr,
            'max_iters': max_iters,
            'n_head': int(self.variant['embed_dim'] // 128)
        })

    def _initialize_environment(self):
        self.state_dim, self.act_dim, self.action_range = get_env_spec(self.variant["env"])
        if "antmaze" in self.variant["env"] and self.variant["conditioning"] == "subgoal":
            self.subgoal_dim = 2
            self.condition_dim = self.subgoal_dim
        else:
            self.subgoal_dim = self.state_dim
            self.condition_dim = 1
        self.device = self.variant.get("device", "cuda")
        self.target_entropy = -self.act_dim

    def _initialize_model(self):
        self.policy = create_model(self, MAX_EPISODE_LEN)
        self.qf = None
        initialize_q_network(self)
        self.q_loss_mean = get_q_loss_mean(self)

    def _initialize_dataset(self):
        self.offline_trajs, self.state_mean, self.state_std, self.max_return = load_dataset(self.variant["env"], self.variant["conditioning"])
        if "antmaze" in self.variant["env"]:
            self.state_mean = np.zeros_like(self.state_mean)
            self.state_std = np.ones_like(self.state_std)
            
    def _initialize_training_components(self):
        # q aid
        self.policy_optimizer = create_optimizer(self.policy, self.variant["learning_rate"], self.variant["weight_decay"])
        self.policy_scheduler = create_scheduler(self.policy_optimizer, self.variant["warmup_steps"])

        self.trainer_params = {
            "env_name": self.variant["env"],
            "policy": self.policy,
            "policy_optimizer": self.policy_optimizer,
            "policy_scheduler": self.policy_scheduler,
            "device": self.device,
            "qf": self.qf,
            "q_loss_mean": self.q_loss_mean,
            "q_scale": self.variant["q_scale"],
            "min_q": self.variant["min_q"],
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "conditioning": self.variant["conditioning"]
        }

        self.common_dataloader_params = {
            "batch_size": self.variant["batch_size"],
            "max_len": self.variant["K"],
            "state_dim": self.state_dim,
            "subgoal_dim": self.subgoal_dim,
            "act_dim": self.act_dim,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "reward_scale": self.reward_scale,
            "action_range": self.action_range
        }

    def _initialize_logging(self):
        self.logger = Logger(self.variant)

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.policy.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.policy)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        return outputs, None


    def __call__(self):
        
        if self.variant.get('conditioning') == 'subgoal':
            # Check for Antmaze domain when 'subgoal' conditioning is used
            if 'antmaze' not in self.variant["env"]:
                raise RuntimeError('Only the Antmaze domain can use subgoal conditioning.')
        
        if self.variant['log_to_wb']:
            wandb_name = f"{self.variant['env']}"
            group_name = f"{self.variant['base_arch']} ({self.variant['conditioning']})"
            
            env = self.variant['env']
            project_name = 'QCS'
            wandb.init(
                name=wandb_name,
                group=group_name,
                project=project_name,
                config=self.variant
            )

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )
    
        main_trainer = Maintrainer(self)
        main_trainer.train(eval_envs, self.variant['log_to_wb'], self.max_return)

        eval_envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")

    # model options
    parser.add_argument('--base_arch', type=str, default='dc')
    parser.add_argument("--conditioning", type=str, default='rtg', help='rtg or subgoal')
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=8)
    parser.add_argument("--use_action", action='store_true')
    parser.add_argument("--ordering", type=int, default=0)
    
    parser.add_argument("--q_scale", type=float, default=1.0)
    parser.add_argument("--min_q", type=int, default=0.0)

    # shared evaluation options
    parser.add_argument("--num_eval_episodes", type=int, default=10)

    # shared training options
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--num_updates_per_iter", type=int, default=1000)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp/seq")
    parser.add_argument("--exp_name", type=str, default="default")
    
    # DC convolution
    parser.add_argument('--conv_window_size', type=int, default=4)
    
    parser.add_argument("--iql_discount", type=float, default=0.99)
    parser.add_argument("--iql_expectile", type=float, default=0.7)
    parser.add_argument("--iql_layernorm", default=False, action='store_true')
    parser.add_argument("--iql_q_hiddens", type=int, default=2)
    parser.add_argument("--iql_v_hiddens", type=int, default=2)

    args = parser.parse_args()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args))

    print("=" * 50)
    experiment()
