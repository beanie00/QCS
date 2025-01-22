"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from the following:
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

Both are licensed under the MIT License.
"""

import torch
import torch.nn as nn

import transformers

from models.decision_transformer.trajectory_gpt2 import GPT2Model


class DecisionTransformer(nn.Module):

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
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        if use_condition:
            self.embed_condition = nn.Linear(self.condition_dim, hidden_size)
        if use_action:
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.predict_action = nn.Sequential(
            *(
                [nn.Linear(hidden_size, self.act_dim)]
                + ([nn.Tanh()] if action_tanh else [])
            )
        )

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.action_range = action_range

    def forward(
        self,
        states,
        conditions,
        actions,
        timesteps,
        ordering,
        padding_mask=None,
    ):

        batch_size, seq_length = states.shape[:2]
        embeddings = []
        num_modals = 0

        if padding_mask is None:
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)  # Defaults to allow all attention

        # ordering embeddings
        order_embeddings = self.embed_ordering(timesteps) if self.ordering else 0

        # condition embeddings
        if self.use_condition:
            conditions_embeddings = self.embed_condition(conditions) + order_embeddings
            embeddings.append(conditions_embeddings)
            num_modals += 1

        # state embeddings
        state_embeddings = self.embed_state(states) + order_embeddings
        embeddings.append(state_embeddings)
        num_modals += 1

        # action embeddings
        if self.use_action:
            action_embeddings = self.embed_action(actions) + order_embeddings
            embeddings.append(action_embeddings)
            num_modals += 1

        stacked_inputs = torch.stack(embeddings, dim=1).permute(0, 2, 1, 3).reshape(batch_size, num_modals * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Prepare the padding mask
        repeated_padding_masks = [padding_mask for _ in range(num_modals)]
        stacked_padding_mask = torch.stack(repeated_padding_masks, dim=1).permute(0, 2, 1).reshape(batch_size, num_modals * seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]
        x = x.reshape(batch_size, seq_length, num_modals, self.hidden_size).permute(0, 2, 1, 3)

        # predict next action given state
        if self.use_condition:
            action_preds = self.predict_action(x[:, 1])
        else:
            action_preds = self.predict_action(x[:, 0])

        return action_preds

    def get_action_predictions(
        self, states, conditions, actions, timesteps, num_envs=1
    ):
        states = states.reshape(num_envs, -1, self.state_dim)
        conditions = conditions.reshape(num_envs, -1, self.condition_dim)
        actions = actions.reshape(num_envs, -1, self.act_dim)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            conditions = conditions[:, -self.eval_context_length :]
            actions = actions[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

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
        else:
            padding_mask = None

        action_preds = self.forward(
            states,
            conditions,
            actions,
            timesteps,
            ordering,
            padding_mask=padding_mask
        )
        return self.clamp_action(action_preds[:, -1])

    def clamp_action(self, action):
        return action.clamp(*self.action_range)