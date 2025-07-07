import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import time
from frameDisplay import FrameDisplay

# Prioritized Experience Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = list(zip(*samples))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(np.array(batch[1]))
        rewards = torch.FloatTensor(np.array(batch[2]))
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.BoolTensor(np.array(batch[4]))
        
        return states, actions, rewards, next_states, dones, indices, torch.FloatTensor(weights)
    
    def update_priorities(self, indices, priorities):
        # Ensure priorities is always iterable
        if hasattr(priorities, '__len__'):
            priorities = np.array(priorities).flatten()
        else:
            priorities = [priorities]
        
        if hasattr(indices, '__len__'):
            indices = np.array(indices).flatten()
        else:
            indices = [indices]
            
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-6
            
    def __len__(self):
        return len(self.buffer)

# Noisy Linear Layer for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
        
    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(input, weight, bias)

# Rainbow DQN Network
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions, action_history_length=10, n_atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.action_history_length = action_history_length
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # CNN layers for frames
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        conv_out_size = self._get_conv_out(input_shape)
        
        # Action history embedding
        self.action_embed = nn.Embedding(n_actions, 32)
        
        # Combined features
        combined_size = conv_out_size + (action_history_length * 32)
        
        # Dueling architecture with noisy layers
        self.advantage_hidden = NoisyLinear(combined_size, 512)
        self.advantage_out = NoisyLinear(512, n_actions * n_atoms)
        
        self.value_hidden = NoisyLinear(combined_size, 512)
        self.value_out = NoisyLinear(512, n_atoms)
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, n_atoms))
        
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def forward(self, frames, action_history):
        batch_size = frames.size(0)
        
        # Process frames through CNN
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        # Process action history
        action_embeds = self.action_embed(action_history)
        action_embeds = action_embeds.view(batch_size, -1)
        
        # Combine features
        combined = torch.cat([x, action_embeds], dim=1)
        
        # Advantage stream
        advantage = F.relu(self.advantage_hidden(combined))
        advantage = self.advantage_out(advantage).view(batch_size, self.n_actions, self.n_atoms)
        
        # Value stream
        value = F.relu(self.value_hidden(combined))
        value = self.value_out(value).view(batch_size, 1, self.n_atoms)
        
        # Dueling aggregation
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Softmax for probability distribution
        q_dist = F.softmax(q_atoms, dim=-1)
        
        return q_dist
    
    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()

class RainbowDQNAgent:
    def __init__(self, input_shape, n_actions, action_history_length=10, lr=0.0001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=32, target_update=1000,
                 n_atoms=51, v_min=-10, v_max=10, multi_step=3):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.action_history_length = action_history_length
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.multi_step = multi_step
        
        # Networks
        self.q_network = RainbowDQN(input_shape, n_actions, action_history_length, n_atoms, v_min, v_max).to(self.device)
        self.target_network = RainbowDQN(input_shape, n_actions, action_history_length, n_atoms, v_min, v_max).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.multi_step_buffer = deque(maxlen=multi_step)
        
        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        # Training metrics
        self.step_count = 0
        self.episode_rewards = []
        self.losses = []
        
        # Frame display for visualization
        self.frame_display = FrameDisplay(frame_shape=(64, 96), scale=4, spacing=5, window_size=(640, 480), num_q_values=8)
        
        # Disable matplotlib visualization - keep only emulator display
        # self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        # self.fig.suptitle('Rainbow DQN Training Progress')
        # plt.ion()
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            frames = torch.FloatTensor(state['frames']).unsqueeze(0).to(self.device)
            action_history = torch.LongTensor(state['action_history']).unsqueeze(0).to(self.device)
            q_dist = self.q_network(frames, action_history)
            q_values = (q_dist * self.support).sum(dim=2)
            action = q_values.argmax(dim=1).item()
            
            # Display frames and Q-values
            self.frame_display.display_frames(state['frames'], q_values.squeeze(0))
        
        return action
    
    def _compute_multi_step_return(self, rewards, gamma):
        """Compute n-step return"""
        n_step_return = 0
        for i, reward in enumerate(rewards):
            n_step_return += (gamma ** i) * reward
        return n_step_return
    
    def store_transition(self, state, action, reward, next_state, done):
        self.multi_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.multi_step_buffer) == self.multi_step or done:
            # Compute n-step return
            rewards = [transition[2] for transition in self.multi_step_buffer]
            n_step_return = self._compute_multi_step_return(rewards, self.gamma)
            
            # Get the first state and action, and the last next_state
            first_state, first_action = self.multi_step_buffer[0][:2]
            last_next_state, last_done = self.multi_step_buffer[-1][3:]
            
            self.memory.add(first_state, first_action, n_step_return, last_next_state, last_done)
            
            if done:
                self.multi_step_buffer.clear()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from prioritized replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q distribution
        current_q_dist = self.q_network(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        # Target Q distribution
        with torch.no_grad():
            # Double DQN: use main network to select action, target network to evaluate
            next_q_dist = self.q_network(next_states)
            next_q_values = (next_q_dist * self.support).sum(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            
            target_q_dist = self.target_network(next_states)
            target_q_dist = target_q_dist[range(self.batch_size), next_actions]
            
            # Compute target distribution
            target_support = rewards.unsqueeze(1) + (self.gamma ** self.multi_step) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # Distribute probability
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            target_dist = torch.zeros_like(target_q_dist)
            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)
            
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_q_dist * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_q_dist * (b - l.float())).view(-1))
        
        # Compute loss
        loss_per_sample = -(target_dist * current_q_dist.log()).sum(dim=1)
        
        # Update priorities with individual TD errors
        td_errors = loss_per_sample.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Apply importance sampling weights and reduce to scalar loss
        loss = (loss_per_sample * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Reset noise
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.losses.append(loss.item())
        
    def update_visualization(self, episode_reward, episode):
        self.episode_rewards.append(episode_reward)
        
        # Print progress to console instead of showing plots
        if len(self.episode_rewards) % 10 == 0:  # Print every 10 episodes
            recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Last 10 = {recent_avg:.2f}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            
            if len(self.losses) > 0:
                recent_loss = np.mean(self.losses[-10:]) if len(self.losses) >= 10 else self.losses[-1]
                print(f"  Recent Loss: {recent_loss:.4f}")
        
        # Keep the emulator display frame visible by not interfering with it
    
    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
        
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
