"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from pathlib import Path
import os
import random
import torch
import numpy as np

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def snapshot_src(src, target, exclude_from):
    mkdir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")

class Logger:
    def __init__(self, variant):

        self.log_path = self.create_log_path(variant)
        mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")

    def log_metrics(self, outputs, iter_num, total_transitions_sampled, writer):
        print("=" * 80)
        print(f"Iteration {iter_num}")
        for k, v in outputs.items():
            print(f"{k}: {v}")
            if writer:
                writer.add_scalar(k, v, iter_num)
                if k == "evaluation/return_mean_gm":
                    writer.add_scalar(
                        "evaluation/return_vs_samples",
                        v,
                        total_transitions_sampled,
                    )

    def create_log_path(self, variant):
        env_name = variant["env"]
        seed = variant["seed"]
        prefix = variant["save_dir"]
        return f"{prefix}/{env_name}-{variant['base_arch']}-{variant['conditioning']}-seed-{seed}"

def save_model(self, path_prefix):
    postfix = 'model'
    to_save = {
        "policy_state_dict": self.policy.state_dict(),
        "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
        "policy_scheduler_state_dict": self.policy_scheduler.state_dict(),
        "iter": self.iter,
        "args": self.variant,
        "np": np.random.get_state(),
        "python": random.getstate(),
        "pytorch": torch.get_rng_state(),
    }
    with open(f"{path_prefix}/{postfix}.pt", "wb") as f:
        torch.save(to_save, f)
    print(f"Model saved at {path_prefix}/{postfix}.pt")


def load_model(self, path_prefix, is_pretrain_model=False):
    postfix = 'pretrain_model' if is_pretrain_model else 'model'
    if Path(f"{path_prefix}/{postfix}.pt").exists():
        with open(f"{path_prefix}/{postfix}.pt", "rb") as f:
            checkpoint = torch.load(f)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
        self.iter = checkpoint["iter"]
        np.random.set_state(checkpoint["np"])
        random.setstate(checkpoint["python"])
        torch.set_rng_state(checkpoint["pytorch"])
        print(f"Model loaded at {path_prefix}/{postfix}.pt")