import torch
import torch.nn as nn


class MLPActor(nn.Module):

    """
    Simple MLP that predicts next action a from past states s and conditions c.
    """

    def __init__(
        self,
        state_dim,
        condition_dim,
        act_dim,
        hidden_size,
        action_range,
        n_layer,
        use_condition=True,
        use_action=False,
        dropout=0.1,
        max_length=1,
        **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.use_condition = use_condition
        self.use_action = use_action
        self.action_range = action_range
        self.max_length = max_length

        if self.use_condition:
            input_dim = max_length*self.state_dim + max_length*self.condition_dim
        else:
            input_dim = max_length*self.state_dim

        layers = [nn.Linear(input_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        states,
        conditions,
        actions=None,
        timesteps=None,
        ordering=None,
        padding_mask=None
    ):
        states = states[:,-self.max_length:].reshape(states.shape[0], -1)
        conditions = conditions[:,-self.max_length:].reshape(conditions.shape[0], -1)
        if self.use_condition:
            inputs = torch.cat((states, conditions), dim=1)
        else:
            inputs = states
            
        actions = self.model(inputs).reshape(inputs.shape[0], 1, self.act_dim)
        return actions

    def get_action_predictions(
        self, states, conditions, actions=None, timesteps=None, num_envs=1
    ):
        states = states.reshape(num_envs, -1, self.state_dim).to(dtype=torch.float32)
        conditions = conditions.reshape(num_envs, -1, self.condition_dim).to(dtype=torch.float32)
        
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((num_envs, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
            conditions = torch.cat(
                [torch.zeros((num_envs, self.max_length-conditions.shape[1], self.condition_dim),
                             dtype=torch.float32, device=conditions.device), conditions], dim=1)
        
        actions = self.forward(states, conditions, None)
        return actions[:,-1]