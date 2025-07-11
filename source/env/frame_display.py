import cv2
import numpy as np
from collections import deque
import json
import os
import csv
import time


class FrameDisplay:
    def __init__(self, frame_shape=(64, 96), scale=3, spacing=5, window_size=(640, 480), num_actions=10):
        self.frame_shape = frame_shape
        self.scale = scale
        self.spacing = spacing
        self.window_size = window_size
        self.num_actions = num_actions
        self.bar_height = 80  
        
        self.reward_history = deque(maxlen=1000)
        self.loss_history = deque(maxlen=1000)
        self.episode_rewards = deque(maxlen=500)
        self.episode_numbers = deque(maxlen=500)
        self.current_episode = 0
        self.step_count = 0
        
        os.makedirs('metrics', exist_ok=True)
        self.step_metrics_file = 'metrics/step_metrics.csv'
        self.episode_metrics_file = 'metrics/episode_metrics.csv'
        
        self._init_csv_files()
        
        self.action_names = [
            "None",           # 0: Do nothing
            "Walk->",          # 1: Walk right
            "Run->",           # 2: Run right
            "Jump->",         # 3: Quick jump right
            "Hold3->",        # 4: Hold jump right (3 frames)
            "Hold4->",        # 5: Hold jump right (4 frames)
            "Hold5->",        # 6: Hold jump right (5 frames) - tall jumps
            "RunJump3->",    # 7: Running jump (3 frames)
            "RunJump5->",    # 8: Long running jump (5 frames) - tall jumps
            "Walk<-",        # 9: Walk left (for backing up)
        ]


    def _init_csv_files(self):
        if not os.path.exists(self.step_metrics_file):
            with open(self.step_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'step', 'episode', 'reward', 'loss'])
        
        if not os.path.exists(self.episode_metrics_file):
            with open(self.episode_metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'episode', 'total_reward', 'max_x_pos', 'death_reason', 'level_completed', 'duration'])


    def display_frames(self, array, action_probs, current_reward=None, current_loss=None):
        if array.shape[1:] != self.frame_shape:
            raise ValueError(f"Each frame in the array must be of shape {self.frame_shape}")
        if len(action_probs) != self.num_actions:
            raise ValueError(f"Number of action probabilities must be {self.num_actions}")

        if current_reward is not None:
            self._log_step_data(current_reward, current_loss)

        frames = self._prepare_frames(array)
        resized_frames = [self._resize_frame(frame, self.scale) for frame in frames]
        concatenated_frames = self._concatenate_frames(resized_frames, self.spacing)

        frame_width = concatenated_frames.shape[1]
        action_probs_frame = self._create_action_probabilities_bar(action_probs, frame_width)
        final_frame = np.vstack([concatenated_frames, action_probs_frame])
        final_frame = self._center_in_window(final_frame, self.window_size)

        cv2.imshow('PPO Mario DS - Frame Stack with Action Probabilities', final_frame)
        cv2.waitKey(1)


    def _prepare_frames(self, array):
        array = array.astype(np.float32)
        array -= array.min()
        array /= array.max()
        array *= 255
        array = array.astype(np.uint8)

        return [frame for frame in array]

    def _resize_frame(self, frame, scale):
        height, width = frame.shape
        return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_NEAREST)

    def _concatenate_frames(self, frames, spacing):
        height = max(frame.shape[0] for frame in frames)
        total_width = sum(frame.shape[1] for frame in frames) + spacing * (len(frames) - 1)
        concatenated = np.zeros((height, total_width), dtype=np.uint8)

        current_x = 0
        for frame in frames:
            concatenated[:frame.shape[0], current_x:current_x + frame.shape[1]] = frame
            current_x += frame.shape[1] + spacing

        return concatenated

    def _center_in_window(self, frame, window_size):
        window_height, window_width = window_size
        frame_height, frame_width = frame.shape[:2]

        vertical_padding = max((window_height - frame_height) // 2, 0)
        horizontal_padding = max((window_width - frame_width) // 2, 0)

        return cv2.copyMakeBorder(frame, vertical_padding, vertical_padding, 
                                  horizontal_padding, horizontal_padding, 
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    def _create_action_probabilities_bar(self, action_probs, width):
        if hasattr(action_probs, 'cpu'):
            action_probs = action_probs.cpu().numpy()
        
        max_prob = np.max(action_probs)
        if max_prob > 0:
            probs_normalized = (action_probs / max_prob) * (self.bar_height - 20)
        else:
            probs_normalized = np.zeros_like(action_probs)

        bar_width = int(width / self.num_actions)
        probs_frame = np.zeros((self.bar_height, width), dtype=np.uint8)

        max_action = np.argmax(action_probs)

        for i, (prob, norm_value) in enumerate(zip(action_probs, probs_normalized)):
            left = i * bar_width
            right = min(left + bar_width - 2, width - 1)  
            top = self.bar_height - int(norm_value) - 15  
            
            bar_color = 200 if i == max_action else 120
            cv2.rectangle(probs_frame, (left + 2, top), (right, self.bar_height - 15), bar_color, -1)
            
            prob_text = f"{prob*100:.0f}%"
            text_size = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            text_x = max(left + (bar_width - text_size[0]) // 2, 0)
            text_y = max(top - 2, 10)
            if text_x + text_size[0] < width:  
                cv2.putText(probs_frame, prob_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            action_name = self.action_names[i] if i < len(self.action_names) else f"A{i}"
            name_size = cv2.getTextSize(action_name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            name_x = max(left + (bar_width - name_size[0]) // 2, 0)
            name_y = self.bar_height - 3
            
            text_color = (255, 255, 255) if i == max_action else (180, 180, 180)
            if name_x + name_size[0] < width:  
                cv2.putText(probs_frame, action_name, (name_x, name_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

        return probs_frame


    def update_episode_data(self, episode_info):
        episode_number = episode_info.get('episode', self.current_episode)
        episode_reward = episode_info.get('total_reward', 0)
        
        self.episode_rewards.append(episode_reward)
        self.episode_numbers.append(episode_number)
        self.current_episode = episode_number

        self._log_episode_data(episode_info)
        self._save_episode_snapshot(episode_info)
    

    def _log_step_data(self, step_reward, loss=None):
        self.step_count += 1
        self.reward_history.append(step_reward)
        if loss is not None:
            self.loss_history.append(loss)
        
        timestamp = time.time()
        with open(self.step_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.step_count, self.current_episode, step_reward, loss])
    

    def _log_episode_data(self, episode_info):
        timestamp = time.time()
        with open(self.episode_metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                episode_info.get('episode', self.current_episode),
                episode_info.get('total_reward', 0),
                episode_info.get('max_x_position', 0),
                episode_info.get('death_reason', 'unknown'),
                episode_info.get('level_completed', False),
                episode_info.get('episode_duration', 0)
            ])
    

    def _save_episode_snapshot(self, episode_info):
        snapshot = {
            'timestamp': time.time(),
            'episode': episode_info.get('episode', self.current_episode),
            'episode_info': episode_info,
            'recent_rewards': list(self.reward_history)[-50:],  
            'recent_losses': list(self.loss_history)[-50:],     
            'episode_rewards': list(self.episode_rewards),
            'episode_numbers': list(self.episode_numbers)
        }
        
        snapshot_file = f'metrics/episode_{episode_info.get("episode", self.current_episode)}_snapshot.json'
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        with open('metrics/latest_snapshot.json', 'w') as f:
            json.dump(snapshot, f, indent=2)


    def get_metrics_summary(self):
        return {
            'current_episode': self.current_episode,
            'total_steps': self.step_count,
            'recent_avg_reward': np.mean(list(self.reward_history)[-10:]) if len(self.reward_history) > 0 else 0,
            'recent_avg_loss': np.mean(list(self.loss_history)[-10:]) if len(self.loss_history) > 0 else 0,
            'episode_count': len(self.episode_rewards),
            'avg_episode_reward': np.mean(list(self.episode_rewards)) if len(self.episode_rewards) > 0 else 0
        }
    

    def close(self):
        summary = self.get_metrics_summary()
        with open('metrics/final_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        cv2.destroyAllWindows()
