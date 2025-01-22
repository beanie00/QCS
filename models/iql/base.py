import os
import copy
import torch
import wandb

from models.iql.value import QNetwork, ValueNetwork


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        expectile=0.7,
        discount=0.99,
        tau=0.005,
        hidden_dim=256,
        q_hiddens=2,
        v_hiddens=2,
        layernorm=False,
    ):
        self.qf = QNetwork(state_dim, action_dim, hidden_dim, q_hiddens, layernorm).to(device)
        self.qf_target = copy.deepcopy(self.qf)
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=3e-4)

        self.vf = ValueNetwork(state_dim, hidden_dim, v_hiddens, layernorm).to(device)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.expectile = expectile

    def update_v(self, states, actions, log_to_wb=False):
        with torch.no_grad():
            q1, q2 = self.qf_target(states, actions)
            q = torch.minimum(q1, q2).detach()

        v = self.vf(states)
        value_loss = loss(q - v, self.expectile).mean()

        self.vf_optimizer.zero_grad()
        value_loss.backward()
        self.vf_optimizer.step()

        if log_to_wb:
            logs = dict()
            logs["IQL training/state_value_loss"] = value_loss
            logs["IQL training/state_value"] = v.mean()
            wandb.log(logs, step=self.total_it)

    def update_q(self, states, actions, rewards, next_states, not_dones, log_to_wb=False):
        with torch.no_grad():
            next_v = self.vf(next_states)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        q1, q2 = self.qf(states, actions)
        q_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        self.qf_optimizer.zero_grad()
        q_loss.backward()
        self.qf_optimizer.step()

        if log_to_wb:
            logs = dict()
            logs["IQL training/q_loss"] = q_loss
            logs["IQL training/q1"] = q1.mean()
            logs["IQL training/q2"] = q2.mean()
            wandb.log(logs, step=self.total_it)

    def update_target(self):
        for param, target_param in zip(self.qf.parameters(), self.qf_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, replay_buffer, batch_size=256, log_to_wb=False):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Update
        self.update_v(state, action, log_to_wb)
        self.update_q(state, action, reward, next_state, not_done, log_to_wb)
        self.update_target()

    def save(self, model_dir):
        torch.save(self.qf.state_dict(), os.path.join(model_dir, f"qf_{str(self.total_it)}.pth"))
        torch.save(self.qf_target.state_dict(), os.path.join(model_dir, f"qf_target_{str(self.total_it)}.pth"))
        torch.save(self.qf_optimizer.state_dict(), os.path.join(
            model_dir, f"qf_optimizer_{str(self.total_it)}.pth"))

        torch.save(self.vf.state_dict(), os.path.join(model_dir, f"vf_{str(self.total_it)}.pth"))
        torch.save(self.vf_optimizer.state_dict(), os.path.join(
            model_dir, f"vf_optimizer_{str(self.total_it)}.pth"))