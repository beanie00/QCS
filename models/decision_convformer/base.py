import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import TanhTransform
from torch import distributions as pyd

import transformers
from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
)
from transformers.utils import logging
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from models.decision_convformer.convolution import Convolution


# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        hidden_size = config.n_embd

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.conv = Convolution(config, hidden_size, config.use_condition, config.use_action)

        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        block_activation = nn.GELU() if config.activation_function == 'gelu' else nn.ReLU()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, config.mlp_expansion*hidden_size),
            block_activation,
            nn.Linear(config.mlp_expansion*hidden_size, hidden_size),
            nn.Dropout(config.drop_p),
        )

        self.index = index

    def forward(
            self,
            hidden_states,
    ):
        conv_output = self.conv(self.ln_1(hidden_states))
        hidden_states = conv_output + hidden_states

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))

        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change



class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, index) for index in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def forward(self, inputs_embeds=None):
        hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)

        for i, block in enumerate(self.h):
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class DecisionConvFormer(nn.Module):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            condition_dim,
            act_dim,
            hidden_size,
            action_range,
            use_condition=True,
            use_action=False,
            ordering=0,
            max_length=None,
            eval_context_length=None,
            max_ep_len=4096,
            action_tanh=True,
            mlp_expansion=4,
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.use_condition = use_condition
        self.use_action = use_action

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            use_condition=use_condition,
            use_action=use_action,
            mlp_expansion=mlp_expansion,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        if self.use_condition:
            self.embed_condition = torch.nn.Linear(self.condition_dim, hidden_size)
        if self.use_action:
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.ordering = ordering
        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )
        self.eval_context_length = eval_context_length
        self.action_range = action_range

    def forward(self, states, conditions, actions, timesteps, ordering, padding_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        # embed each modality with a different head
        embeddings = []
        num_modals = 0

        if self.use_condition:
            conditions_embeddings = self.embed_condition(conditions) + order_embeddings
            embeddings.append(conditions_embeddings)
            num_modals += 1

        state_embeddings = self.embed_state(states) + order_embeddings
        embeddings.append(state_embeddings)
        num_modals += 1

        if self.use_action:
            action_embeddings = self.embed_action(actions) + order_embeddings
            embeddings.append(action_embeddings)
            num_modals += 1

        # Stack and reshape the inputs
        stacked_inputs = torch.stack(embeddings, dim=1).permute(0, 2, 1, 3).reshape(batch_size, num_modals * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        x = self.transformer(inputs_embeds=stacked_inputs)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, num_modals, self.hidden_size).permute(0, 2, 1, 3)

        # predict next action given state
        if self.use_condition:
            preds = self.predict_action(x[:, 1])
        else:
            preds = self.predict_action(x[:, 0])

        return preds


    def get_action_predictions(
        self, states, conditions, actions, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)
        conditions = conditions.reshape(num_envs, -1, self.condition_dim)
        actions = actions[:, -self.eval_context_length :]
        timesteps = timesteps[:, -self.eval_context_length :]

        ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction

        states = states[:, -self.eval_context_length :]
        conditions = conditions[:, -self.eval_context_length :]

        states = torch.cat(
            [
                torch.zeros(
                    (
                        states.shape[0],
                        self.max_length - states.shape[1],
                        self.state_dim,
                    ),
                    device=states.device,
                ),
                states,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        conditions = torch.cat(
            [
                torch.zeros(
                    (
                        conditions.shape[0],
                        self.max_length - conditions.shape[1],
                        self.condition_dim,
                    ),
                    device=conditions.device,
                ),
                conditions,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)

        timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

        ordering = torch.cat(
            [
                torch.zeros(
                    (ordering.shape[0], self.max_length - ordering.shape[1]),
                    device=ordering.device,
                ),
                ordering,
            ],
            dim=1,
        ).to(dtype=torch.long)

        action_preds = self.forward(
            states,
            conditions,
            actions,
            timesteps,
            ordering,
            **kwargs
        )
        return self.clamp_action(action_preds[:, -1])

    def clamp_action(self, action):
        return action.clamp(*self.action_range)