import random
import gym
from gym import spaces
import numpy as np
import time
from desmume.emulator import DeSmuME, DeSmuME_Savestate
from pympler.tracker import SummaryTracker

import os

from moviepy.editor import ImageSequenceClip

from source.env.data_processing import preprocess_image
from source.env.input import Input

valid_saves = [
    "saves/W1-3.sav",
]

def create_episode_video(images, episode_number):
    output_path = f'episodes/episode_{episode_number}.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = [np.array(img) for img in images]
    clip = ImageSequenceClip(frames, fps=60)
    clip.write_videofile(output_path, codec='libx264', audio=False, bitrate='50M')

    return output_path

tracker = SummaryTracker()

class MarioDSEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, frame_skip=5, frame_stack=4, ppo_optimized=True):
        super(MarioDSEnv, self).__init__()

        self.ppo_optimized = ppo_optimized
        if ppo_optimized:
            self.action_space = spaces.Discrete(10)  # Back to 10 actions with just one left movement
            self.observation_space = spaces.Box(low=0, high=1, shape=(frame_stack, 64, 96), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(8) 
            self.action_history_length = 10
            self.observation_space = spaces.Dict({
                'frames': spaces.Box(low=0, high=1, shape=(frame_stack, 64, 96), dtype=np.float32),
                'action_history': spaces.Box(low=0, high=7, shape=(10,), dtype=np.int32)
            })
        self.frame_skip = frame_skip
        self.frame_stack_num = frame_stack

        self.emu = DeSmuME()
        self.emu.open('NSMB.nds')
        self.window = self.emu.create_sdl_window()
        self.saver = DeSmuME_Savestate(self.emu)
        self.saver.load_file(valid_saves[random.randint(0, len(valid_saves) - 1)])

        self.frame_count = 0
        self.inputs = Input(self.emu)
        
        if ppo_optimized:
            self.action_mapping = {
                0: self.inputs.none,                    # Do nothing
                1: self.inputs.walk_right,              # Walk right (main direction)
                2: self.inputs.run_right,               # Run right (faster movement)
                3: self.inputs.jump_right,              # Quick jump right (1 frame)
                4: self.inputs.hold_jump_right_short,   # Short sustained jump (3 frames)
                5: self.inputs.hold_jump_right_medium,  # Medium sustained jump (4 frames)  
                6: self.inputs.hold_jump_right_long,    # Long sustained jump (5 frames) - for tall obstacles
                7: self.inputs.run_jump_right,          # Running jump (3 frames)
                8: self.inputs.run_jump_right_long,     # Long running jump (5 frames) - for tall obstacles
                9: self.inputs.walk_left,               # Walk left (for backing up when needed)
            }
        else:
            # Original DQN action mapping
            self.action_mapping = {
                0: self.inputs.none,
                1: self.inputs.walk_left,
                2: self.inputs.walk_right,
                3: self.inputs.run_left,
                4: self.inputs.run_right,
                5: self.inputs.jump,
                6: self.inputs.jump_left,
                7: self.inputs.jump_right
            }
        
        self.emu.volume_set(0)
        self.frame_stack = np.zeros(((self.frame_skip * self.frame_stack_num), 64, 96), dtype=np.float32)
        self.episode_frames = []
        
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.step_count = 0
        self.episode_start_time = 0
        self.stuck_count = 0
        self.last_progress_step = 0
        self.episode_reward = 0
        self.episode_actions = []
        self.death_reason = "unknown"

        if not ppo_optimized:
            self.action_history = [0] * self.action_history_length
        
        self.reset()


    def step(self, action):
        self.current_stack = self.frame_stack
        self.action_mapping[action]()
        self.inputs.execute_action()

        if not self.ppo_optimized:
            self.action_history.append(action)
            self.action_history.pop(0)  # Remove the oldest action

        self.emu.cycle()
        self.frame_count += 1
        self.step_count += 1

        frame = self.emu.screenshot()
        self.episode_frames.append(frame)
        dead = False
        
        self.frame_array, dead = preprocess_image(frame)
            
        self.frame_stack = np.concatenate((self.frame_stack, self.frame_array.reshape(1, 64, 96))) 
        self.frame_stack = self.frame_stack[1:, :, :]  

        current_x_pos = self.emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000
        reward = current_x_pos - 0.01
        
        if current_x_pos > self.max_x_pos:
            self.max_x_pos = current_x_pos
            self.last_progress_step = self.step_count
            self.stuck_count = 0
        else:
            self.stuck_count += 1
        
        self.prev_x_pos = current_x_pos
        
        self.episode_reward += reward
        self.episode_actions.append(action)
        
        if dead:
            self.death_reason = "enemy_or_pit"
        elif self.frame_count > 3000:
            self.death_reason = "timeout"
            
        if dead or self.frame_count > 3000:
            dead = True
            reward = -1

        info = {"errors": "No errors"}  

        frame_skip_frames = np.zeros((self.frame_stack_num, 64, 96), dtype=np.float32)

        for i in range(self.frame_stack_num):
            index = -1 - (self.frame_skip * i)
            frame_skip_frames[self.frame_stack_num - 1 - i] = self.frame_stack[index]

        if self.ppo_optimized:
            return frame_skip_frames, reward, dead, False, info
        else:
            observation = {
                'frames': frame_skip_frames,
                'action_history': np.array(self.action_history, dtype=np.int32)
            }
            return observation, reward, dead, False, info


    def reset(self, save_movie=False, episode=None):
        if save_movie:
            create_episode_video(self.episode_frames, episode)
        
        self.episode_frames = []
        self.frame_count = 0
        self.step_count = 0
        self.episode_start_time = time.time()

        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.stuck_count = 0
        self.last_progress_step = 0
        self.episode_reward = 0
        self.episode_actions = []
        self.death_reason = "unknown"

        if not self.ppo_optimized:
            self.action_history = [0] * self.action_history_length

        self.saver.load_file(valid_saves[random.randint(0, len(valid_saves) - 1)])
        self.frame_stack = np.zeros(((self.frame_skip * self.frame_stack_num), 64, 96), dtype=np.float32)
        self.state = None 

        if self.ppo_optimized:
            return np.zeros((self.frame_stack_num, 64, 96), dtype=np.float32)
        else:
            initial_observation = {
                'frames': np.zeros((self.frame_stack_num, 64, 96), dtype=np.float32),
                'action_history': np.array(self.action_history, dtype=np.int32)
            }
            return initial_observation


    def get_episode_info(self):
        return {
            'max_x_position': self.max_x_pos,
            'final_x_position': self.prev_x_pos,
            'episode_duration': time.time() - self.episode_start_time,
            'stuck_count': self.stuck_count,
            'last_progress_step': self.last_progress_step,
            'total_frames': len(self.episode_frames),
            'total_reward': self.episode_reward,
            'total_actions': len(self.episode_actions),
            'action_distribution': {i: self.episode_actions.count(i) for i in range(self.action_space.n)},
            'death_reason': self.death_reason,
            'level_completed': self.max_x_pos > 0.8,  
            'average_progress_per_step': self.max_x_pos / max(self.step_count, 1)
        }


    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.emu.screenshot()
        elif mode == 'human':
            self.window.draw()


    def close(self):
        self.emu.destroy()
        self.window.destroy()
        
