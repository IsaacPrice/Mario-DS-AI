import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict
import time
from source.env.frame_display import FrameDisplay
from source.ppo.memory import PPOMemory
from source.ppo.network import PPONetwork


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
        
        self.policy = PPONetwork(input_shape, n_actions).to(self.device)
        self.memory = PPOMemory()
        self.frame_display = FrameDisplay(frame_shape=(64, 96), scale=3, spacing=5, window_size=(640, 480), num_actions=10)
        
        self.logger = None 
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.steps_done = 0
        self.update_count = 0
        self.current_step_reward = 0
        self.current_loss = 0
        self.action_counts = defaultdict(int)
        self.episode_action_counts = defaultdict(int)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)


    def select_action(self, state):
        with torch.no_grad():
            frames = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.act(frames)
            
            self.action_counts[action] += 1
            self.episode_action_counts[action] += 1
            
            action_probs, _ = self.policy.forward(frames)
            self.frame_display.display_frames(
                state, 
                action_probs.squeeze(0), 
                current_reward=self.current_step_reward, 
                current_loss=self.current_loss if self.current_loss > 0 else None
            )
            
        return action, log_prob.item(), value.item()
    

    def act(self, state, training=True):
        if training:
            return self.select_action(state)
        else:
            action, _, _ = self.select_action(state)
            return action
    

    def compute_gae(self, rewards, values, dones, next_value, gamma=None, lam=None):
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
        
        frames, actions, rewards, dones, old_log_probs, old_values = self.memory.get_tensors(self.device)
        
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
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        for epoch in range(self.k_epochs):
            action_probs, values = self.policy(frames)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            entropy_loss = -entropy
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        avg_policy_loss = total_policy_loss / self.k_epochs
        avg_value_loss = total_value_loss / self.k_epochs
        avg_entropy_loss = total_entropy_loss / self.k_epochs
        avg_total_loss = total_loss / self.k_epochs
        
        update_time = time.time() - update_start_time
        self.update_count += 1
        
        self.losses.append(avg_total_loss)
        self.current_loss = avg_total_loss
        
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
        self.current_step_reward = reward


    def end_episode(self, episode_info):
        episode_reward = episode_info.get('total_reward', 0)
        episode_length = episode_info.get('total_actions', 0)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        episode_info['episode'] = len(self.episode_rewards)
        self.frame_display.update_episode_data(episode_info)
        
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
        
        self.episode_action_counts.clear()
        

    def set_logger(self, logger):
        self.logger = logger
        

    def update_visualization(self, episode_reward, episode):
        self.episode_rewards.append(episode_reward)
        
        if len(self.episode_rewards) % 10 == 0:  # Print every 10 episodes
            recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Last 10 = {recent_avg:.2f}")
            
            if len(self.losses) > 0:
                recent_loss = np.mean(self.losses[-10:]) if len(self.losses) >= 10 else self.losses[-1]
                print(f"  Recent Loss: {recent_loss:.4f}")
        
        
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
