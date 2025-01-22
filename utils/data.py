"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import torch
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

def plot_antmaze_pos(trajectories):
    selected_trajectories = random.sample(trajectories, 500)
    for path in selected_trajectories:
        states = path["observations"][:, :2]
        plt.scatter(states[:, 0], states[:, 1], s=3, color=np.random.rand(3,))
    plt.savefig('tsne.png')
    

MAX_EPISODE_LEN = 1000


class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        subgoal_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.subgoal_dim = subgoal_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood. We clamp them to be within
        # the user defined action range.
        self.action_range = action_range

    def __call__(self, traj):
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        ss = traj["observations"][si : si + self.max_len].reshape(-1, self.state_dim)
        sg = traj["subgoals"][si : si + self.max_len].reshape(-1, self.subgoal_dim)
        aa = traj["actions"][si : si + self.max_len].reshape(-1, self.act_dim)
        rr = traj["rewards"][si : si + self.max_len].reshape(-1, 1)
        tr = traj["traj_returns"][si : si + self.max_len].reshape(-1, 1)
        if "terminals" in traj:
            dd = traj["terminals"][si : si + self.max_len]  # .reshape(-1)
        else:
            dd = traj["dones"][si : si + self.max_len]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = ss.shape[0]

        timesteps = np.arange(si, si + tlen)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff

        rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        act_len = aa.shape[0]
        if tlen != act_len:
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), ss])
        ss = (ss - self.state_mean) / self.state_std

        sg = np.concatenate([np.zeros((self.max_len - tlen, self.subgoal_dim)), sg])

        aa = np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa])
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr])
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 1, dd])
        tr = np.concatenate([np.zeros((self.max_len - tlen, 1)), tr])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg])
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps])
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])

        ss = torch.from_numpy(ss).to(dtype=torch.float32)
        sg = torch.from_numpy(sg).to(dtype=torch.float32)
        aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*self.action_range)
        rr = torch.from_numpy(rr).to(dtype=torch.float32)
        dd = torch.from_numpy(dd).to(dtype=torch.long)
        tr = torch.from_numpy(tr).to(dtype=torch.float32)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long)
        ordering = torch.from_numpy(ordering).to(dtype=torch.long)
        padding_mask = torch.from_numpy(padding_mask)

        return ss, sg, aa, rr, dd, tr, rtg, timesteps, ordering, padding_mask


def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    subgoal_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    num_workers=24
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size)

    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        subgoal_dim=subgoal_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
    )

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)

    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size):

    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)

    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds

def load_dataset(env_name, conditioning):
    env_name_list = env_name.split('-')
    if len(env_name_list) == 4 and env_name_list[2] == 'expert':
        dataset_path = f'data/{env_name_list[0]}-expert-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        dataset_path = f'data/{env_name_list[0]}-medium-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories += pickle.load(f)
        random.shuffle(trajectories)
    else:
        dataset_path = f'data/{env_name}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    
    # plot_antmaze_pos(trajectories)
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        traj_return = path["rewards"].sum()
        returns.append(traj_return)
        
        path["traj_returns"] = np.array([traj_return for i in range(len(path["rewards"]))])
        
        if conditioning == "subgoal":
            # random subgoals
            num_states = path["observations"].shape[0]
            path_subgoals = []
            for i in range(num_states-1):
                random_idx = np.random.randint(i+1, num_states)
                path_subgoals.append(path["observations"][random_idx][:2])
            path_subgoals.append(path["observations"][-1][:2])
            path["subgoals"] = np.array(path_subgoals)
        else:
            # set dummy subgoals to avoid errors (not used)
            path["subgoals"] = path["observations"]

    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
    print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
    print("=" * 50)

    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    trajectories = [trajectories[ii] for ii in sorted_inds]

    return trajectories, state_mean, state_std, np.max(returns)