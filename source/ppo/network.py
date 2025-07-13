import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import numpy as np


class PPONetwork(nn.Module):
    def __init__(self, input_shape, n_actions=6, hidden_size=512, binary_mode=True):
        super(PPONetwork, self).__init__()
        
        self.binary_mode = binary_mode
        self.n_actions = n_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.shared_fc = nn.Linear(conv_out_size, hidden_size)
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        
        if binary_mode:
            self.actor_out = nn.Linear(hidden_size, 6)  # [UP, DOWN, LEFT, RIGHT, X, A]
        else:
            self.actor_out = nn.Linear(hidden_size, n_actions)
        
        self.critic_out = nn.Linear(hidden_size, 1)
        

    def _get_conv_out(self, shape):
        o: torch.Tensor = torch.zeros(1, *shape)
        o = self.features(o)
        return int(np.prod(o.size()))
    

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.features(frames)
        x = x.view(x.size(0), -1)

        shared: torch.Tensor = F.relu(self.shared_fc(x))
        actor: torch.Tensor = F.relu(self.actor_fc(shared))

        if self.binary_mode:
            action_probs: torch.Tensor = torch.sigmoid(self.actor_out(actor))
        else:
            action_probs: torch.Tensor = F.softmax(self.actor_out(actor), dim=1)

        critic: torch.Tensor = F.relu(self.critic_fc(shared))
        value: torch.Tensor = self.critic_out(critic)

        return action_probs, value
    
    
    def act(self, frames):
        action_probs, value = self.forward(frames)
        
        if self.binary_mode:
            dists = [Bernoulli(prob) for prob in action_probs.squeeze()]
            actions = [dist.sample() for dist in dists]
            log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
            
            binary_actions = [action.item() for action in actions]
            total_log_prob = sum(log_probs)
            
            return binary_actions, total_log_prob, value
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob, value
    
    