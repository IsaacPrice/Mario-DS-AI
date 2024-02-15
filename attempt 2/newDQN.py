import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import os
from layers import NoisyLinear, ConvBlock

class DistributionalDuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, num_bins=51):
        super(DistributionalDuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.num_bins = num_bins

        # Convolutional layers (same as before)
        self.conv_layers = nn.Sequential(
            ConvBlock(input_shape[0], 32, kernel_size=8, stride=4),
            ConvBlock(32, 64, kernel_size=4, stride=2),
            ConvBlock(64, 64, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Common part of the Dueling Architecture
        self.common = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU()
        )

        # Modified value stream to output a distribution
        self.value_stream = nn.Sequential(
            NoisyLinear(512, self.num_bins),
        )

        # Modified advantage stream to output a distribution for each action
        self.advantage_stream = nn.Sequential(
            NoisyLinear(512, 512, std_init=1),
            nn.ReLU(),
            NoisyLinear(512, n_actions * self.num_bins, std_init=1),
        )

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        conv_out = self.conv_layers(x)
        common = self.common(conv_out)

        # Compute value and advantage distributions
        value = self.value_stream(common).view(-1, 1, self.num_bins)
        advantage = self.advantage_stream(common).view(-1, self.n_actions, self.num_bins)

        # Combine value and advantage streams
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get probability distributions
        q_vals = nn.functional.softmax(q_vals, dim=2)

        return q_vals

    def reset_noise(self):
        for layer in self.common:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.value_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class DQN:
    def __init__(self, input_shape, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, num_bins=51, batch_size=32, Vmax=10, Vmin=-10):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_bins = num_bins
        self.Vmax = Vmax
        self.Vmin = Vmin

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.q_network = DistributionalDuelingDQN(input_shape, n_actions, num_bins).to(self.device)
        self.target_network = DistributionalDuelingDQN(input_shape, n_actions, num_bins).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss = nn.KLDivLoss()

        self.memory = []
        self.mem_size = 100000
        self.mem_cntr = 0

        self.learn_step = 0
        self.replace_target = 100

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).to(self.device).float().unsqueeze(0)
                distribution = self.q_network(state_tensor)
                expected_values = distribution.mean(dim=2).squeeze(0)  # Calculate expected values
                action = torch.argmax(expected_values).item()
        return action
        
    def learn(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state).to(self.device).float().unsqueeze(0)
        next_state_tensor = torch.tensor(next_state).to(self.device).float().unsqueeze(0)

        current_distribution = self.q_network(state_tensor).squeeze(0)
        next_distribution = self.target_network(next_state_tensor).mean(dim=2).squeeze(0)
        next_action = torch.argmax(next_distribution)
        next_distribution = self.target_network(next_state_tensor).squeeze(0)[next_action]

        target_distribution = self._compute_target_distribution(next_distribution, reward, done)

        self.optimizer.zero_grad()
        # Ensure current_distribution is indexed correctly
        # If current_distribution is [num_actions, num_bins]
        current_q_distribution = current_distribution[action]

        # Computing the loss
        # Ensure target_distribution is aligned in shape with current_q_distribution
        loss = -torch.sum(target_distribution * torch.log(current_q_distribution + 1e-8))
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _compute_target_distribution(self, next_distribution, reward, done):
        delta_z = (self.Vmax - self.Vmin) / (self.num_bins - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_bins).to(self.device)

        # Initialize the target distribution with zeros
        target_distribution = torch.zeros_like(next_distribution)

        # Bellman update with clipping for a single reward and done signal
        for i in range(self.num_bins):
            tz = reward + self.gamma * support[i] * (1 - done)
            tz = torch.clamp(tz, self.Vmin, self.Vmax)
            b = (tz - self.Vmin) / delta_z

            # Compute lower and upper indices
            lower = b.floor().long()
            upper = b.ceil().long()

            # Ensure indices are within bounds
            lower.clamp_(0, self.num_bins - 1)
            upper.clamp_(0, self.num_bins - 1)

            # Calculate the contribution of each bin
            lower_contribution = (upper.float() - b) * next_distribution[i]
            upper_contribution = (b - lower.float()) * next_distribution[i]

            # Distribute the probabilities
            target_distribution[lower] += lower_contribution
            target_distribution[upper] += upper_contribution

        return target_distribution

    
    def hard_update(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
    
    def reset_noise(self):
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def save(self, filename="dqn_checkpoint.pth"):
        checkpoint = {
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'loss': self.loss,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load(self, filename="dqn_checkpoint.pth"):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.loss = checkpoint['loss']
            self.epsilon = checkpoint['epsilon']
            print(f"Checkpoint loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}")

