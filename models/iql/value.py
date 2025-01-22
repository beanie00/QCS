import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layers(in_dim, hidden_dim, n_hiddens, use_layer_norm):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    for _ in range(n_hiddens-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.Linear(hidden_dim, 1))
    return layers

class ValueNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_hiddens,
        layernorm,
        **kwargs
    ) -> None:
        super().__init__()
        layers = create_layers(in_dim, hidden_dim, n_hiddens, layernorm)
        self.v = nn.Sequential(*layers)

    def forward(self, state):
        return self.v(state)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_hiddens, layernorm):
        super(QNetwork, self).__init__()
        
        if layernorm:
            layers = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm)
            self.q1 = nn.Sequential(*layers)
            self.q2 = nn.Sequential(*layers)
        else:
            layers1 = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm)
            layers2 = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm)
            self.q1 = nn.Sequential(*layers1)
            self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)
        
        return q1, q2