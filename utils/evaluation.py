"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import gym
import numpy as np
import torch
import wandb
from d4rl import infos

MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    vec_env,
    env_name,
    eval_rtg,
    state_dim,
    subgoal_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    device,
    stage,
):
    def eval_episodes_fn(actor):
        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        returns, lengths, _ = vec_evaluate_seq(
            vec_env,
            state_dim,
            subgoal_dim,
            act_dim,
            actor,
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
        )
        suffix = f"{stage} "
        reward_min = infos.REF_MIN_SCORE[env_name]
        reward_max = infos.REF_MAX_SCORE[env_name]
        return {
            f"{suffix}evaluation/{int(eval_rtg)} return_mean": np.mean(returns),
            # f"{suffix}evaluation/{int(eval_rtg)} return_std": np.std(returns),
            # f"{suffix}evaluation/{int(eval_rtg)} length_mean": np.mean(lengths),
            # f"{suffix}evaluation/{int(eval_rtg)} length_std": np.std(lengths),
            f"{suffix}evaluation/{int(eval_rtg)} d4rl_score": (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),
        }

    return eval_episodes_fn


def vec_evaluate_seq(
    vec_env,
    state_dim,
    subgoal_dim,
    act_dim,
    actor,
    target_return: list,
    value_buffer=None,
    value_update=None,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    total_transitions_sampled=0,
):

    actor.eval()
    actor.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()
    
    if subgoal_dim == 2:
        target_goal = np.array(vec_env.get_attr("target_goal"))
    
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    subgoals = torch.empty(num_envs, 0, subgoal_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    state = torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim).to(dtype=torch.float32)
    prev_state = states
    
    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )

    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )
        
        with torch.no_grad():
            if subgoal_dim == 2:
                subgoal = (
                    torch.from_numpy(target_goal)
                    .reshape(num_envs, subgoal_dim)
                    .to(device=device, dtype=torch.float32)
                ).reshape(num_envs, -1, subgoal_dim)
                subgoals = torch.cat([subgoals, (subgoal)], dim=1)
                conditions= subgoals
            else:
                conditions = target_return.to(dtype=torch.float32)
            action = actor.get_action_predictions(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                conditions,
                actions.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                num_envs=num_envs,
            )
            
        action = action.clamp(*actor.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())
        
        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
        
        torch_state = torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        torch_reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
                
        actions[:, -1] = action
        states = torch.cat([states, (torch_state)], dim=1)
        rewards[:, -1] = torch_reward
        
        if mode != "delayed":
            pred_return = target_return[:, -1] - (torch_reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "next_observations": states[ii].detach().cpu().numpy()[1:ep_len+1],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )