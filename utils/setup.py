import torch
from models.decision_transformer.base import DecisionTransformer
from models.decision_convformer.base import DecisionConvFormer
from models.mlp_actor import MLPActor
from utils import Lamb, create_vec_eval_episodes_fn

def create_model(self, MAX_EPISODE_LEN):
    if 'dt' in self.variant['base_arch']:
        return DecisionTransformer(
            env_name=self.variant['env'],
            state_dim=self.state_dim,
            condition_dim=self.condition_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            use_condition=self.variant['base_arch'] != 'dt-no-condition',
            use_action=self.variant["use_action"],
            max_length=self.variant["K"],
            eval_context_length=self.variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=self.variant["embed_dim"],
            n_layer=self.variant["n_layer"],
            n_head=self.variant["n_head"],
            n_inner=4 * self.variant["embed_dim"],
            activation_function=self.variant["activation_function"],
            n_positions=1024,
            resid_pdrop=self.variant["dropout"],
            attn_pdrop=self.variant["dropout"],
            ordering=self.variant["ordering"],
            plot_attention=self.variant['plot_attention']
        ).to(device=self.device)
    if 'dc' in self.variant['base_arch']:
        return DecisionConvFormer(
            env_name=self.variant['env'],
            state_dim=self.state_dim,
            condition_dim=self.condition_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            use_condition=self.variant['base_arch'] != 'dc-no-condition',
            use_action=self.variant["use_action"],
            max_length=self.variant["K"],
            eval_context_length=self.variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=self.variant["embed_dim"],
            n_layer=self.variant["n_layer"],
            n_head=self.variant["n_head"],
            n_inner=4 * self.variant["embed_dim"],
            activation_function=self.variant["activation_function"],
            drop_p=self.variant["dropout"],
            ordering=self.variant["ordering"],
            window_size=self.variant['conv_window_size']
        ).to(device=self.device)
        
    if 'mlp' in self.variant['base_arch']:
        return MLPActor(
            state_dim=self.state_dim,
            condition_dim=self.condition_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            hidden_size=self.variant["embed_dim"],
            n_layer=self.variant["n_layer"],
            use_condition=self.variant['base_arch'] != 'mlp-no-condition',
            use_action=self.variant["use_action"],
            dropout=self.variant["dropout"],
            max_length=self.variant["K"],
        ).to(device=self.device)

def create_optimizer(model, learning_rate, weight_decay):
    return Lamb(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8
    )

def create_scheduler(optimizer, warmup_steps):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps)

def create_eval_function(self, eval_envs, eval_rtg, stage):
    return create_vec_eval_episodes_fn(
        vec_env=eval_envs,
        env_name=self.variant["env"],
        eval_rtg=eval_rtg,
        state_dim=self.state_dim,
        subgoal_dim=self.subgoal_dim,
        act_dim=self.act_dim,
        state_mean=self.state_mean,
        state_std=self.state_std,
        reward_scale=self.reward_scale,
        device=self.device,
        stage=stage
    )
