"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
import torch.nn.functional as F
from d4rl import infos

EXP_ADV_MAX = 100.

class SequenceTrainer:
    def __init__(
        self,
        env_name,
        policy,
        policy_optimizer,
        policy_scheduler,
        qf,
        q_loss_mean,
        state_mean,
        state_std,
        conditioning,
        q_scale,
        min_q,
        device="cuda",
    ):
        self.env_name = env_name
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.policy_scheduler = policy_scheduler
        
        self.qf = qf
        self.q_loss_mean = q_loss_mean
        self.state_mean = state_mean
        self.state_std = state_std
        self.conditioning = conditioning
        self.q_scale = q_scale
        self.min_q = min_q
        
        self.device = device
        self.start_time = time.time()

    def train_iteration(
        self,
        dataloader
    ):

        losses, action_losses, q_losses, weighted_q_losses = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.policy.train()
        for i, trajs in enumerate(dataloader):
            loss, action_loss, q_loss, weighted_q_loss = self.train_step(trajs)
            losses.append(loss)
            action_losses.append(action_loss)
            q_losses.append(q_loss)
            weighted_q_losses.append(weighted_q_loss)

        logs["time/training"] = time.time() - train_start
        logs["training/action_loss_mean"] = np.mean(action_losses)
        logs["training/action_loss_std"] = np.std(action_losses)
        logs["training/q_loss_mean"] = np.mean(q_losses)
        logs["training/q_loss_std"] = np.std(q_losses)
        logs["training/loss_mean"] = np.mean(losses)
        logs["training/loss_std"] = np.std(losses)
        logs["training/weighted_q_loss_mean"] = np.mean(weighted_q_losses)
        logs["training/weighted_q_loss_std"] = np.std(weighted_q_losses)

        return logs

    def train_step(self, trajs):
        (
            states,
            subgoals,
            actions,
            _,
            _,
            traj_returns,
            rtgs,
            timesteps,
            ordering,
            padding_mask,
        ) = [tensor.to(self.device) for tensor in trajs]

        action_target = actions.clone()
        state_mean_tensor = torch.from_numpy(self.state_mean).to(actions.device)
        state_std_tensor = torch.from_numpy(self.state_std).to(actions.device)

        if "antmaze" in self.env_name and self.conditioning == "subgoal":
            conditions = subgoals
        else:
            conditions = rtgs[:, :-1]
            
        # Predict actions
        action_preds = self.policy.forward(
            states, conditions, actions, timesteps, ordering, padding_mask=padding_mask
        )

        # Compute action loss and update the act model
        batch_size, context_len = action_preds.shape[0], action_preds.shape[1]
        act_dim = action_preds.shape[2]
        action_loss = F.mse_loss(action_preds, action_target, reduction="none")
        action_loss = action_loss.mean(dim=2)
        
        action_preds_flat = action_preds.view(-1, act_dim)
        states_flat = states.view(-1, states.shape[2])*state_std_tensor + state_mean_tensor
        q1, q2 = self.qf(states_flat.to(dtype=torch.float32), action_preds_flat.to(dtype=torch.float32))
        q_loss = -torch.minimum(q1, q2).view(batch_size, -1)

        reward_min = infos.REF_MIN_SCORE[f"{self.env_name}"]
        reward_max = infos.REF_MAX_SCORE[f"{self.env_name}"]
        
        normalized_returns = (traj_returns - reward_min) / (reward_max - reward_min)
        if "halfcheetah" in self.env_name:
            normalized_max_return = 0.9
        elif "hopper" in self.env_name or "walker2d" in self.env_name:
            normalized_max_return = 1.1
        else:
            normalized_max_return = 1.0
            
        q_alpha = self.q_scale * (normalized_max_return - normalized_returns) * 100
        q_alpha = torch.clamp(q_alpha, min=self.min_q).view(batch_size, -1)
        lmbda = q_alpha/self.q_loss_mean
        weighted_q_loss = lmbda * q_loss
            
        loss = action_loss + weighted_q_loss
        loss = loss.view(-1, 1)[padding_mask.reshape(-1) > 0].mean()
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.policy_optimizer.step()
        self.policy_scheduler.step()
        
        return loss.item(), action_loss.mean().item(), q_loss.mean().item(), weighted_q_loss.mean().item()
