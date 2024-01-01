import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, num_bins=51):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        # Calculate the total number of input features after flattening
        self.num_features = 12288
        #for dim in input_shape:
        #    self.num_features *= dim

        # Common layers
        self.common = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Outputs a single value
        )

        # Advantage stream
        output_size = n_actions * num_bins
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)  # Outputs a value for each action
        )

    def forward(self, x):
        x = self.common(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q-values
        # Subtract mean advantage to stabilize learning
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class DQN:
    def __init__(self, input_shape, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, num_bins=51, batch_size=32):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.q_network = DuelingDQN(input_shape, n_actions, num_bins).to(self.device)
        self.target_network = DuelingDQN(input_shape, n_actions, num_bins).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        self.memory = []
        self.mem_size = 100000
        self.mem_cntr = 0

        self.learn_step = 0
        self.replace_target = 100

    def choose_action(self, state):
        """
        Choose an action to take given the current state.
        
        state: the current state of the environment
        returns: the action to take
        """
        if np.random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).to(self.device).float().unsqueeze(0)
                distribution = self.target_network.forward(state_tensor)
                return distribution
    
    def learn(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state).to(self.device).float().unsqueeze(0)
        current_q_values = self.q_network.forward(state_tensor)
        current_q = current_q_values[0, action]

        # Compute target Q values
        state_tensor = torch.tensor(next_state).to(self.device).float().unsqueeze(0)
        next_q = self.q_network.forward(state_tensor)
        next_q = next_q.max()
        target_q = reward + self.gamma * next_q

        # Compute loss
        loss = self.loss(current_q, target_q)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def hard_update(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

