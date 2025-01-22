import torch
import torch.nn as nn

class Convolution(nn.Module):
    def __init__(self, config, hidden_size, use_condition, use_action):
        super().__init__()
        self.window_size = config.window_size
        self.use_condition = use_condition
        self.use_action = use_action

        if use_condition:
            self.rtg_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        self.obs_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)
        if use_action:
            self.act_conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=self.window_size, groups=hidden_size)

    def forward(self, x):
        window_size = self.window_size

        # pad the input tensor with zeros along the sequence dimension
        padded_tensor = torch.nn.functional.pad(x, (0, 0, window_size - 1, 0)).transpose(1, 2)

        if self.use_action:
            if self.use_condition:
                rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::3]
                obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::3]
                act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 2::3]

                conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3), act_conv_tensor.unsqueeze(3)), dim=3)
                conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)
            else:
                obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, ::2]
                act_conv_tensor = self.act_conv1d(padded_tensor)[:, :, 1::2]

                conv_tensor = torch.cat((obs_conv_tensor.unsqueeze(3), act_conv_tensor.unsqueeze(3)), dim=3)
                conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)

        else:
            if self.use_condition:
                rtg_conv_tensor = self.rtg_conv1d(padded_tensor)[:, :, ::2]
                obs_conv_tensor = self.obs_conv1d(padded_tensor)[:, :, 1::2]

                conv_tensor = torch.cat((rtg_conv_tensor.unsqueeze(3), obs_conv_tensor.unsqueeze(3)), dim=3)
                conv_tensor = conv_tensor.reshape(conv_tensor.shape[0], conv_tensor.shape[1], -1)
            else:
                conv_tensor = self.obs_conv1d(padded_tensor)

        conv_tensor = conv_tensor.transpose(1, 2)

        return conv_tensor