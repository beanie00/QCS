import os
import torch
import gym
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def initialize_q_network(self):
    from models.iql.value import QNetwork
    self.qf = QNetwork(self.state_dim, self.act_dim, 256, self.variant["iql_q_hiddens"], self.variant["iql_layernorm"]).to(self.device)
    model_dir_path = os.path.join('exp/iql', self.variant["env"], str(self.variant["seed"]))
    full_file_path = os.path.join(model_dir_path, 'qf_1000000.pth')
    
    if not os.path.isdir(model_dir_path):
        raise FileNotFoundError(f"Directory does not exist: {model_dir_path}")
    if not os.path.isfile(full_file_path):
        raise FileNotFoundError(f"File does not exist: {full_file_path}")

    self.qf.load_state_dict(torch.load(full_file_path))
    print(f"Model loaded from {full_file_path}")
    
def get_q_loss_mean(self):
    env = gym.make(self.variant["env"])
    ds = env.get_dataset()
    obs = ds['observations']
    actions = ds['actions']

    dataset = TensorDataset(torch.Tensor(obs), torch.Tensor(actions))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    tqdm_bar = tqdm(dataloader)
    
    total_q_loss = 0
    for batch_idx, (obs, act) in enumerate(tqdm_bar):
        batch_loss = 0

        obs = obs.cuda()
        act = act.cuda()

        q1, q2 = self.qf(obs, act)
        q_loss = torch.minimum(q1, q2).mean()
        
        batch_loss = q_loss.item() 
        total_q_loss += q_loss.item() 

        tqdm_bar.set_description('Q Loss: {:.2g}'.format(batch_loss))

    return abs(total_q_loss / (batch_idx + 1))