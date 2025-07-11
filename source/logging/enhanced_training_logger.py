import json
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

class EnhancedTrainingLogger:
    def __init__(self, log_dir="training_logs", experiment_name=None):
        if experiment_name is None:
            experiment_name = f"mario_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.episode_data = []
        self.training_data = []
        self.start_time = time.time()
        self.reward_window = deque(maxlen=100)
        self.length_window = deque(maxlen=100)
        self.progress_window = deque(maxlen=100)
        self.best_reward = float('-inf')
        self.best_progress = 0
        self.episodes_completed = 0
        self.total_steps = 0
        
        self._initialize_csv_files()
        self._save_config()
        
        print(f"Enhanced Training Logger initialized: {self.experiment_name}")
        print(f"Logs will be saved to: {self.log_dir}")
    
    def _initialize_csv_files(self):
        episode_headers = [
            'episode', 'reward', 'length', 'max_x_position', 'level_completed',
            'avg_reward_100', 'avg_length_100', 'avg_progress_100', 
            'best_reward_so_far', 'best_progress_so_far', 'elapsed_time_minutes',
            'steps_per_second', 'action_distribution', 'death_reason'
        ]
        
        with open(os.path.join(self.log_dir, 'episode_data.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_headers)
        
        training_headers = [
            'update_step', 'policy_loss', 'value_loss', 'entropy_loss', 'total_loss',
            'learning_rate', 'clip_ratio', 'kl_divergence', 'explained_variance',
            'grad_norm', 'update_time_seconds'
        ]
        
        with open(os.path.join(self.log_dir, 'training_data.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(training_headers)
    
    def _save_config(self):
        config = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'log_directory': self.log_dir,
            'description': 'Enhanced PPO training with multi-frame actions for tall jumps'
        }
        
        with open(os.path.join(self.log_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_episode(self, episode, reward, length, max_x_pos, level_completed=False, 
                   action_counts=None, death_reason="timeout"):
        
        current_time = time.time()
        elapsed_minutes = (current_time - self.start_time) / 60
        
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.progress_window.append(max_x_pos)
        
        if reward > self.best_reward:
            self.best_reward = reward
        if max_x_pos > self.best_progress:
            self.best_progress = max_x_pos
        
        avg_reward = np.mean(self.reward_window)
        avg_length = np.mean(self.length_window)
        avg_progress = np.mean(self.progress_window)
        self.total_steps += length
        steps_per_second = self.total_steps / (current_time - self.start_time)
        
        action_dist_str = ""
        if action_counts:
            total_actions = sum(action_counts.values())
            action_dist = {k: f"{v/total_actions*100:.1f}%" for k, v in action_counts.items()}
            action_dist_str = json.dumps(action_dist)
        
        episode_data = {
            'episode': episode,
            'reward': reward,
            'length': length,
            'max_x_position': max_x_pos,
            'level_completed': level_completed,
            'avg_reward_100': avg_reward,
            'avg_length_100': avg_length,
            'avg_progress_100': avg_progress,
            'best_reward_so_far': self.best_reward,
            'best_progress_so_far': self.best_progress,
            'elapsed_time_minutes': elapsed_minutes,
            'steps_per_second': steps_per_second,
            'action_distribution': action_dist_str,
            'death_reason': death_reason
        }
        
        self.episode_data.append(episode_data)
        
        with open(os.path.join(self.log_dir, 'episode_data.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, length, max_x_pos, level_completed,
                avg_reward, avg_length, avg_progress,
                self.best_reward, self.best_progress, elapsed_minutes,
                steps_per_second, action_dist_str, death_reason
            ])
        
        self._print_episode_summary(episode_data)
        
        if episode % 50 == 0:
            self.save_plots()
    
    def log_training_update(self, update_step, policy_loss, value_loss, entropy_loss, 
                          total_loss, learning_rate=None, clip_ratio=None, 
                          kl_divergence=None, explained_variance=None, 
                          grad_norm=None, update_time=None):
        training_data = {
            'update_step': update_step,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'learning_rate': learning_rate,
            'clip_ratio': clip_ratio,
            'kl_divergence': kl_divergence,
            'explained_variance': explained_variance,
            'grad_norm': grad_norm,
            'update_time_seconds': update_time
        }
        
        self.training_data.append(training_data)
        
        with open(os.path.join(self.log_dir, 'training_data.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                update_step, policy_loss, value_loss, entropy_loss, total_loss,
                learning_rate, clip_ratio, kl_divergence, explained_variance,
                grad_norm, update_time
            ])
    
    def _print_episode_summary(self, data):
        episode = data['episode']
        reward = data['reward']
        length = data['length']
        progress = data['max_x_position']
        completed = data['level_completed']
        avg_reward = data['avg_reward_100']
        elapsed = data['elapsed_time_minutes']
        sps = data['steps_per_second']
        
        status = "COMPLETED" if completed else "❌ Failed"
        
        print(f"\n{'='*80}")
        print(f"EPISODE {episode:4d} | {status}")
        print(f"{'='*80}")
        print(f"Reward: {reward:8.2f} | Length: {length:4d} steps | Progress: {progress:6.0f}")
        print(f"Avg Reward (100): {avg_reward:6.2f} | Best Reward: {data['best_reward']:6.2f}")
        print(f"Best Progress: {data['best_progress']:6.0f} | Time: {elapsed:5.1f}m | SPS: {sps:4.1f}")
        
        if data['action_distribution']:
            print(f"Actions: {data['action_distribution']}")
        
        print(f"Death: {data['death_reason']}")
        print(f"{'='*80}")
    
    def save_plots(self):
        if len(self.episode_data) < 10:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Mario DS PPO Training Progress - {self.experiment_name}', fontsize=16)
        
        episodes = [data['episode'] for data in self.episode_data]
        rewards = [data['reward'] for data in self.episode_data]
        lengths = [data['length'] for data in self.episode_data]
        progress = [data['max_x_position'] for data in self.episode_data]
        avg_rewards = [data['avg_reward_100'] for data in self.episode_data]
        avg_progress = [data['avg_progress_100'] for data in self.episode_data]
        
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        axes[0, 0].plot(episodes, avg_rewards, color='red', linewidth=2, label='Avg (100 ep)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(episodes, lengths, alpha=0.6, color='green')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(episodes, progress, alpha=0.3, color='purple', label='Max Progress')
        axes[0, 2].plot(episodes, avg_progress, color='orange', linewidth=2, label='Avg (100 ep)')
        axes[0, 2].set_title('Level Progress')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Max X Position')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        if self.training_data:
            updates = [data['update_step'] for data in self.training_data]
            policy_losses = [data['policy_loss'] for data in self.training_data]
            value_losses = [data['value_loss'] for data in self.training_data]
            total_losses = [data['total_loss'] for data in self.training_data]

            axes[1, 0].plot(updates, policy_losses, label='Policy Loss', color='red')
            axes[1, 0].plot(updates, value_losses, label='Value Loss', color='blue')
            axes[1, 0].plot(updates, total_losses, label='Total Loss', color='black')
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        completed_episodes = [data['level_completed'] for data in self.episode_data]
        window_size = min(50, len(completed_episodes))
        if window_size > 0:
            success_rates = []
            for i in range(window_size, len(completed_episodes) + 1):
                window = completed_episodes[i-window_size:i]
                success_rate = sum(window) / len(window) * 100
                success_rates.append(success_rate)
            
            axes[1, 1].plot(episodes[window_size-1:], success_rates, color='green', linewidth=2)
            axes[1, 1].set_title(f'Success Rate ({window_size} episode window)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].axis('off')
        summary_text = f"""
Training Summary:
Episodes: {len(self.episode_data)}
Best Reward: {self.best_reward:.2f}
Best Progress: {self.best_progress:.0f}
Avg Reward (last 100): {np.mean(list(self.reward_window)):.2f}
Total Training Time: {(time.time() - self.start_time)/3600:.1f}h
Steps per Second: {self.total_steps/(time.time() - self.start_time):.1f}
        """

        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {self.log_dir}/training_progress.png")
    
    def save_final_summary(self):
        summary = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_data),
            'total_training_time_hours': (time.time() - self.start_time) / 3600,
            'best_reward': self.best_reward,
            'best_progress': self.best_progress,
            'final_avg_reward': np.mean(list(self.reward_window)),
            'final_avg_progress': np.mean(list(self.progress_window)),
            'total_steps': self.total_steps,
            'average_steps_per_second': self.total_steps / (time.time() - self.start_time),
            'completed_episodes': sum(1 for d in self.episode_data if d['level_completed']),
            'completion_rate': sum(1 for d in self.episode_data if d['level_completed']) / len(self.episode_data) * 100
        }
        
        with open(os.path.join(self.log_dir, 'final_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.save_plots()
        
        print(f"\nTRAINING COMPLETE!")
        print(f"All logs saved to: {self.log_dir}")
        print(f"Final summary: {summary}")
