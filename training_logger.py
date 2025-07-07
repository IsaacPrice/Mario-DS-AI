import json
import csv
import os
from datetime import datetime
import numpy as np

class TrainingLogger:
    """
    Backend logging system for training data without visualization.
    Saves comprehensive training metrics and statistics.
    """
    
    def __init__(self, agent_type, save_dir="training_logs", save_interval=10):
        self.agent_type = agent_type
        self.save_dir = save_dir
        self.save_interval = save_interval
        
        # Create logging directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data storage
        self.episode_data = []
        self.training_data = []
        self.session_start = datetime.now()
        
        # Create session-specific files
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{agent_type}_{timestamp}"
        self.episode_log_file = os.path.join(save_dir, f"{self.session_id}_episodes.csv")
        self.training_log_file = os.path.join(save_dir, f"{self.session_id}_training.csv")
        self.summary_file = os.path.join(save_dir, f"{self.session_id}_summary.json")
        
        # Initialize CSV files with headers
        self._initialize_csv_files()
        
        print(f"Training logger initialized for {agent_type}")
        print(f"Session ID: {self.session_id}")
        print(f"Logs will be saved to: {save_dir}")
    
    def _initialize_csv_files(self):
        """Initialize CSV files with appropriate headers"""
        # Episode log headers
        episode_headers = [
            'episode', 'reward', 'steps', 'timestamp', 'epsilon_or_loss',
            'avg_reward_last_10', 'avg_reward_last_100', 'best_reward_so_far',
            'total_training_time_minutes'
        ]
        
        with open(self.episode_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_headers)
        
        # Training log headers (for detailed training metrics)
        training_headers = [
            'episode', 'update_step', 'loss', 'timestamp', 'additional_metrics'
        ]
        
        with open(self.training_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(training_headers)
    
    def log_episode(self, episode, reward, steps, additional_info=None):
        """Log episode completion data"""
        timestamp = datetime.now()
        training_time = (timestamp - self.session_start).total_seconds() / 60  # minutes
        
        # Calculate statistics
        self.episode_data.append({
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'timestamp': timestamp.isoformat(),
            'additional_info': additional_info or {}
        })
        
        # Calculate averages
        recent_rewards = [ep['reward'] for ep in self.episode_data[-10:]]
        avg_reward_10 = np.mean(recent_rewards) if recent_rewards else 0
        
        recent_rewards_100 = [ep['reward'] for ep in self.episode_data[-100:]]
        avg_reward_100 = np.mean(recent_rewards_100) if recent_rewards_100 else 0
        
        best_reward = max([ep['reward'] for ep in self.episode_data]) if self.episode_data else reward
        
        # Get agent-specific metric (epsilon for DQN, latest loss for PPO)
        agent_metric = additional_info.get('epsilon', additional_info.get('loss', 0)) if additional_info else 0
        
        # Write to CSV
        with open(self.episode_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, steps, timestamp.isoformat(), agent_metric,
                avg_reward_10, avg_reward_100, best_reward, training_time
            ])
        
        # Save summary every save_interval episodes
        if episode % self.save_interval == 0:
            self._save_summary()
            print(f"Episode {episode}: Reward={reward:.2f}, Avg(10)={avg_reward_10:.2f}, "
                  f"Avg(100)={avg_reward_100:.2f}, Best={best_reward:.2f}")
    
    def log_training_step(self, episode, update_step, loss, additional_metrics=None):
        """Log training step data (losses, gradients, etc.)"""
        timestamp = datetime.now()
        
        training_entry = {
            'episode': episode,
            'update_step': update_step,
            'loss': loss,
            'timestamp': timestamp.isoformat(),
            'additional_metrics': additional_metrics or {}
        }
        
        self.training_data.append(training_entry)
        
        # Write to CSV
        with open(self.training_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, update_step, loss, timestamp.isoformat(),
                json.dumps(additional_metrics) if additional_metrics else '{}'
            ])
    
    def _save_summary(self):
        """Save comprehensive summary statistics"""
        if not self.episode_data:
            return
        
        rewards = [ep['reward'] for ep in self.episode_data]
        steps = [ep['steps'] for ep in self.episode_data]
        
        summary = {
            'session_info': {
                'session_id': self.session_id,
                'agent_type': self.agent_type,
                'start_time': self.session_start.isoformat(),
                'last_update': datetime.now().isoformat(),
                'total_episodes': len(self.episode_data)
            },
            'reward_statistics': {
                'total_episodes': len(rewards),
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'median_reward': float(np.median(rewards)),
                'last_10_avg': float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards)),
                'last_100_avg': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
            },
            'episode_statistics': {
                'mean_steps': float(np.mean(steps)),
                'std_steps': float(np.std(steps)),
                'min_steps': int(np.min(steps)),
                'max_steps': int(np.max(steps))
            },
            'training_progress': {
                'recent_performance': rewards[-10:] if len(rewards) >= 10 else rewards,
                'improvement_trend': self._calculate_trend(rewards),
                'training_losses': [t['loss'] for t in self.training_data[-50:]]  # Last 50 training steps
            }
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _calculate_trend(self, rewards, window=20):
        """Calculate if performance is improving, stable, or declining"""
        if len(rewards) < window * 2:
            return "insufficient_data"
        
        early_avg = np.mean(rewards[-window*2:-window])
        recent_avg = np.mean(rewards[-window:])
        
        improvement = (recent_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_statistics(self):
        """Get current training statistics"""
        if not self.episode_data:
            return {}
        
        rewards = [ep['reward'] for ep in self.episode_data]
        return {
            'total_episodes': len(self.episode_data),
            'average_reward': np.mean(rewards),
            'best_reward': max(rewards),
            'recent_average': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            'training_time_minutes': (datetime.now() - self.session_start).total_seconds() / 60
        }
    
    def finalize_session(self):
        """Finalize the training session and save final summary"""
        self._save_summary()
        
        final_stats = self.get_statistics()
        print(f"\nTraining session completed!")
        print(f"Total episodes: {final_stats.get('total_episodes', 0)}")
        print(f"Average reward: {final_stats.get('average_reward', 0):.2f}")
        print(f"Best reward: {final_stats.get('best_reward', 0):.2f}")
        print(f"Training time: {final_stats.get('training_time_minutes', 0):.1f} minutes")
        print(f"Logs saved in: {self.save_dir}")
        
        return final_stats
