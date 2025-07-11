import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import time
from frameDisplay import FrameDisplay
from enhanced_training_logger import EnhancedTrainingLogger

class PPONetwork(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=512):
        super(PPONetwork, self).__init__()
        
        # CNN layers for feature extraction from frames
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        conv_out_size = self._get_conv_out(input_shape)
        
        # Shared features
        self.shared_fc = nn.Linear(conv_out_size, hidden_size)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, n_actions)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_size, hidden_size)
        self.critic_out = nn.Linear(hidden_size, 1)
        
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def forward(self, frames):
        # Feature extraction from frames
        x = F.relu(self.conv1(frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        # Shared features
        shared = F.relu(self.shared_fc(x))
        
        # Actor (policy) output
        actor = F.relu(self.actor_fc(shared))
        action_probs = F.softmax(self.actor_out(actor), dim=1)
        
        # Critic (value) output
        critic = F.relu(self.critic_fc(shared))
        value = self.critic_out(critic)
        
        return action_probs, value
    
    def act(self, frames):
        action_probs, value = self.forward(frames)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_probs = []
        self.old_values = []
    
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.old_log_probs.append(log_prob)
        self.old_values.append(value)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_probs = []
        self.old_values = []
    
    def get_tensors(self, device):
        # States are now simple numpy arrays instead of dictionaries
        frames = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        # Fix the boolean tensor creation issue
        dones_array = np.array(self.dones, dtype=bool)
        dones = torch.tensor(dones_array, dtype=torch.bool).to(device)
        old_log_probs = torch.FloatTensor(self.old_log_probs).to(device)
        old_values = torch.FloatTensor(self.old_values).to(device)
        
        return frames, actions, rewards, dones, old_log_probs, old_values

class PPOAgent:
    def __init__(self, input_shape, n_actions, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
                 update_timestep=4096, gae_lambda=0.95):
        print(torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_timestep = update_timestep
        self.gae_lambda = gae_lambda
        
        # Networks
        self.policy = PPONetwork(input_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training tracking with enhanced logging
        self.logger = None  # Will be set externally
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.steps_done = 0
        self.update_count = 0
        
        # Current step tracking for display
        self.current_step_reward = 0
        self.current_loss = 0
        
        # Action tracking for analysis
        self.action_counts = defaultdict(int)
        self.episode_action_counts = defaultdict(int)
        
        # Frame display for visualization
        self.frame_display = FrameDisplay(frame_shape=(64, 96), scale=3, spacing=5, window_size=(640, 480), num_actions=10)
        
        # Disable matplotlib visualization - keep only emulator display
        # plt.ion()
        # self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        # self.fig.suptitle('PPO Training Progress')
        
    def select_action(self, state):
        with torch.no_grad():
            frames = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.act(frames)
            
            # Track action usage
            self.action_counts[action] += 1
            self.episode_action_counts[action] += 1
            
            # Display frames and action probabilities with current metrics
            action_probs, _ = self.policy.forward(frames)
            self.frame_display.display_frames(
                state, 
                action_probs.squeeze(0), 
                current_reward=self.current_step_reward, 
                current_loss=self.current_loss if self.current_loss > 0 else None
            )
            
        return action, log_prob.item(), value.item()
    
    def act(self, state, training=True):
        """Unified act method for both training and testing"""
        if training:
            return self.select_action(state)
        else:
            # For testing, just return the action
            action, _, _ = self.select_action(state)
            return action
    
    def compute_gae(self, rewards, values, dones, next_value, gamma=None, lam=None):
        """Compute Generalized Advantage Estimation"""
        if gamma is None:
            gamma = self.gamma
        if lam is None:
            lam = self.gae_lambda
            
        values = values + [next_value]
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def update(self):
        if len(self.memory.states) < self.update_timestep:
            return None
            
        update_start_time = time.time()
        
        # Get data from memory
        frames, actions, rewards, dones, old_log_probs, old_values = self.memory.get_tensors(self.device)
        
        # Compute returns and advantages using GAE
        with torch.no_grad():
            _, next_value = self.policy(frames[-1:])
            returns = self.compute_gae(
                rewards.cpu().numpy().tolist(),
                old_values.cpu().numpy().tolist(),
                dones.cpu().numpy().tolist(),
                next_value.cpu().item()
            )
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        for epoch in range(self.k_epochs):
            # Forward pass
            action_probs, values = self.policy(frames)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            entropy_loss = -entropy
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Calculate average losses
        avg_policy_loss = total_policy_loss / self.k_epochs
        avg_value_loss = total_value_loss / self.k_epochs
        avg_entropy_loss = total_entropy_loss / self.k_epochs
        avg_total_loss = total_loss / self.k_epochs
        
        update_time = time.time() - update_start_time
        self.update_count += 1
        
        # Store loss for plotting
        self.losses.append(avg_total_loss)
        # Track current loss for display
        self.current_loss = avg_total_loss
        
        # Log to enhanced logger if available
        if self.logger:
            self.logger.log_training_update(
                update_step=self.update_count,
                policy_loss=avg_policy_loss,
                value_loss=avg_value_loss,
                entropy_loss=avg_entropy_loss,
                total_loss=avg_total_loss,
                learning_rate=self.lr,
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                update_time=update_time
            )
        
        # Clear memory
        self.memory.clear()
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_total_loss,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'update_time': update_time
        }
        
    def store_transition(self, state, action, reward, done, log_prob, value):
        self.memory.store(state, action, reward, done, log_prob, value)
        self.steps_done += 1
        # Track current reward for display
        self.current_step_reward = reward
        
    def end_episode(self, episode_info):
        """Enhanced episode ending with comprehensive logging"""
        episode_reward = episode_info.get('total_reward', 0)
        episode_length = episode_info.get('total_actions', 0)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Update frame display with comprehensive episode data
        episode_info['episode'] = len(self.episode_rewards)
        self.frame_display.update_episode_data(episode_info)
        
        # Log to enhanced logger if available
        if self.logger:
            self.logger.log_episode(
                episode=len(self.episode_rewards),
                reward=episode_reward,
                length=episode_length,
                max_x_pos=episode_info.get('max_x_position', 0),
                level_completed=episode_info.get('level_completed', False),
                action_counts=episode_info.get('action_distribution', {}),
                death_reason=episode_info.get('death_reason', 'unknown')
            )
        
        # Reset episode action counts
        self.episode_action_counts.clear()
        
    def set_logger(self, logger):
        """Set the enhanced training logger"""
        self.logger = logger
        
    def update_visualization(self, episode_reward, episode):
        """Update the training visualization - disabled matplotlib, only console output"""
        self.episode_rewards.append(episode_reward)
        
        # Print progress to console instead of showing plots
        if len(self.episode_rewards) % 10 == 0:  # Print every 10 episodes
            recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Last 10 = {recent_avg:.2f}")
            
            if len(self.losses) > 0:
                recent_loss = np.mean(self.losses[-10:]) if len(self.losses) >= 10 else self.losses[-1]
                print(f"  Recent Loss: {recent_loss:.4f}")
        
        # Keep the emulator display frame visible by not interfering with it
        
    def save_model(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses
        }, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {filepath}")
