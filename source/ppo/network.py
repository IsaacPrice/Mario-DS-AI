import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PPONetwork(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=512):
        super(PPONetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out(input_shape)
        self.shared_fc = nn.Linear(conv_out_size, hidden_size)
        
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, n_actions)
        
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)
        

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    

    def forward(self, frames):
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        shared = F.relu(self.shared_fc(x))
        actor = F.relu(self.actor_fc(shared))
        action_probs = F.softmax(self.actor_out(actor), dim=1)
        critic = F.relu(self.critic_fc(shared))
        value = self.critic_out(critic)
        
        return action_probs, value
    
    
    def act(self, frames):
        action_probs, value = self.forward(frames)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    