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
        self.num_features = 1
        for dim in input_shape:
            self.num_features *= dim

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
        self.num_bins = num_bins
        self.batch_size = batch_size

        # Check if CUDA (GPU support) is available and use it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize and transfer models to the appropriate device
        self.model = DuelingDQN(input_shape, n_actions, num_bins=num_bins).to(self.device)
        self.target_model = DuelingDQN(input_shape, n_actions, num_bins=num_bins).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)

    def compute_expected_values(self, distribution):
        N = self.num_bins  # Number of bins
        bin_edges = torch.linspace(-1, 1, N + 1, device=self.device)  # Adjust range (-1, 1) based on your problem
        bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of each bin

        # Assuming the distribution shape is [batch_size, n_actions * N]
        distribution = distribution.view(-1, self.n_actions, N)

        expected_values = torch.sum(distribution * bin_values, dim=2)
        return expected_values
    
    def compute_target_distribution(self, reward, next_state):
        N = self.num_bins
        bin_edges = torch.linspace(-1, 1, N + 1, device=self.device)
        bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2

        with torch.no_grad():
            next_distribution = self.target_model(next_state)
            next_distribution = next_distribution.view(-1, self.n_actions, N)

            # Compute next state values using double DQN approach
            next_state_values = self.compute_expected_values(next_distribution)
            next_action = torch.argmax(next_state_values, dim=1)
            
            # Compute the target distribution
            Tz = reward + self.gamma * bin_values[next_action]
            Tz = Tz.clamp(min=bin_edges[0], max=bin_edges[-1])
            b = (Tz - bin_edges[0]) / (bin_edges[1] - bin_edges[0])
            l = b.floor().long()
            u = b.ceil().long()

            # Distribute probability
            target_distribution = torch.zeros(next_distribution.shape, device=self.device)
            offset = torch.linspace(0, (self.batch_size - 1) * self.input_shape * N, self.batch_size).long() \
                    .unsqueeze(1).expand(self.batch_size, N).to(self.device)
            
            # Debug
            print(target_distribution.shape)
            print((l + offset).shape)
            print((next_distribution * (u.float() - b)).shape)

            target_distribution.view(-1).index_add_(0, (l + offset).view(-1), (next_distribution * (u.float() - b)).view(-1))
            target_distribution.view(-1).index_add_(0, (u + offset).view(-1), (next_distribution * (b - l.float())).view(-1))

        return target_distribution
 

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                distribution = self.model(state)
                # Compute expected values (mean) from the distributions
                q_values = self.compute_expected_values(distribution)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

        # Get the current distribution
        current_distribution = self.model(state)

        # Compute the target distribution using the distributional Bellman update
        target_distribution = self.compute_target_distribution(reward, next_state)

        # Compute the loss between the current and target distributions
        loss = self.distributional_loss(current_distribution, target_distribution, action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path=''):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """
        Load the model's weights from a given file path.

        :param path: The path to the file containing the model's state dictionary.
        """
        # Check if the file exists
        if not os.path.isfile(path):
            print(f"Error: No file found at '{path}'. Model loading failed.")
            return

        # Load the model's state dictionary
        state_dict = torch.load(path, map_location=self.device)

        # Update the state dictionary of both model and target_model
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)

        print(f"Model loaded successfully from '{path}'.")

    def hard_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self, tau=0.005):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
