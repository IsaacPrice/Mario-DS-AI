import math
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.init import kaiming_uniform_, zeros_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        self.w_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.w_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        self.b_mu = nn.Parameter(torch.empty((out_features)))
        self.b_sigma = nn.Parameter(torch.empty((out_features)))

        kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        kaiming_uniform_(self.w_sigma, a=math.sqrt(5))
        zeros_(self.b_mu)
        zeros_(self.b_sigma)

    def forward(self, x, sigma=0.5):
        if self.training:  # Check if the model is in training mode
            w_noise = torch.normal(0, sigma, size=self.w_mu.size()).to(x.device)
            b_noise = torch.normal(0, sigma, size=self.b_mu.size()).to(x.device)
            return F.linear(x, self.w_mu + self.w_sigma * w_noise, self.b_mu + self.b_sigma * b_noise)
        else:
            return F.linear(x, self.w_mu, self.b_mu)

class DQN(nn.Module):

  def __init__(self, hidden_size, num_layers, obs_shape, n_actions, sigma=0.5, atoms=51): 
    super(DQN, self).__init__()

    self.atoms = atoms
    self.n_actions = n_actions

    self.conv = nn.Sequential(
        ResidualBlock(obs_shape[0], 64, stride=2),
        ResidualBlock(64, 128, stride=2),
        ResidualBlock(128, 256, stride=2)
    )

    conv_out_size = 256 * 6 * 8
    self.head = nn.Sequential()
    for i in range(num_layers):
      self.head.add_module(f'NoisyLinear ({i+1})', NoisyLinear(conv_out_size, hidden_size, sigma=sigma))
      self.head.add_module(f'ReLU ({i+1})', nn.ReLU())
      conv_out_size = hidden_size

    self.fc_adv = NoisyLinear(hidden_size, n_actions * self.atoms, sigma=sigma)
    self.fc_value = NoisyLinear(hidden_size, self.atoms, sigma=sigma)

  def _get_conv_out(self, shape):
    with torch.no_grad():
      conv_out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(conv_out.size()))
    
  def forward(self, x):
    x = x.float()
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    x = self.head(x)
    adv = self.fc_adv(x).view(-1, self.n_actions, self.atoms)
    value = self.fc_value(x).view(-1, 1, self.atoms)
    q_logits = value + adv - adv.mean(dim=1, keepdim=True)
    q_probs = F.softmax(q_logits, dim=-1)
    return q_probs