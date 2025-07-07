import gym
from gym import spaces
import numpy as np
from Input import Input
from DataProccesing import *
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from pympler.tracker import SummaryTracker


import os
import sys
from PIL import Image

from moviepy.editor import ImageSequenceClip

def create_episode_video(images, episode_number):
    output_path = f'episodes/episode_{episode_number}.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert PIL Images to numpy arrays
    frames = [np.array(img) for img in images]

    # Create a video clip from frames
    clip = ImageSequenceClip(frames, fps=60)

    # Write the video file to disk with minimal compression
    clip.write_videofile(output_path, codec='libx264', audio=False, bitrate='50M')

    return output_path

tracker = SummaryTracker()

class MarioDSEnv(gym.Env):
    """
    Custom Environment for Mario DS that follows gym interface.
    This class will be a wrapper around the Nintendo DS emulator to play Mario DS,
    handling state representation, actions, and rewards.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, frame_skip=5, frame_stack=4, ppo_optimized=True):
        super(MarioDSEnv, self).__init__()

        # PPO-optimized configuration
        self.ppo_optimized = ppo_optimized
        
        # Create action and observation space
        if ppo_optimized:
            # Enhanced action space for PPO with sustained jumps for tall obstacles
            self.action_space = spaces.Discrete(10)  # Increased from 6 to 10 actions
            # Enhanced observation space with higher resolution for better platform detection
            self.observation_space = spaces.Box(low=0, high=1, shape=(frame_stack, 64, 96), dtype=np.float32)
        else:
            # Original DQN-style setup  
            self.action_space = spaces.Discrete(8) 
            self.action_history_length = 10
            self.observation_space = spaces.Dict({
                'frames': spaces.Box(low=0, high=1, shape=(frame_stack, 64, 96), dtype=np.float32),
                'action_history': spaces.Box(low=0, high=7, shape=(10,), dtype=np.int32)
            })
        self.frame_skip = frame_skip
        self.frame_stack_num = frame_stack

        # Initialize the emulator and extras
        self.emu = DeSmuME()
        self.emu.open('NSMB.nds')
        self.window = self.emu.create_sdl_window()
        self.saver = DeSmuME_Savestate(self.emu)
        self.saver.load_file('W1-3.sav')

        self.frame_count = 0
        self.inputs = Input(self.emu)
        
        # PPO-optimized action mapping with sustained jumps
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
                9: self.inputs.walk_left,               # Walk left (when needed)
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
        
        # Create the empty frame stack with updated dimensions
        self.emu.volume_set(0)
        self.frame_stack = np.zeros(((self.frame_skip * self.frame_stack_num), 64, 96), dtype=np.float32)
        self.episode_frames = []
        
        # Track previous position for reward calculation
        self.prev_x_pos = 0
        self.max_x_pos = 0
        self.step_count = 0
        
        # Initialize action history (only for DQN mode)
        if not ppo_optimized:
            self.action_history = [0] * self.action_history_length
        
        self.reset()

    def step(self, action):
        """
        Execute one time step within the environment
        """
        # Update the current Framestack
        self.current_stack = self.frame_stack

        # 1. Send the action to the emulator
        self.action_mapping[action]()
        
        # Handle multi-frame actions
        self.inputs.execute_action()

        # Update action history (only for DQN mode)
        if not self.ppo_optimized:
            self.action_history.append(action)
            self.action_history.pop(0)  # Remove the oldest action

        # Move the emulator forward by 1 frame
        self.emu.cycle()
        self.frame_count += 1
        self.step_count += 1

        # 2. Obtain the next state from the emulator
        frame = self.emu.screenshot()
        self.episode_frames.append(frame)
        dead = False
        
        # Use enhanced preprocessing for better platform detection
        if self.ppo_optimized:
            self.frame_array, dead = preprocess_image_ppo_enhanced(frame, width=96, height=64)
        else:
            self.frame_array, dead = preprocess_image_numpy(frame)
            
        self.frame_stack = np.concatenate((self.frame_stack, self.frame_array.reshape(1, 64, 96))) 
        self.frame_stack = self.frame_stack[1:, :, :]  # Remove the oldest frame

        # 3. Calculate the reward with better shaping for PPO
        if self.ppo_optimized:
            reward = self._calculate_ppo_reward(dead)
        else:
            reward = (self.emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000) - 0.01
            
        if dead or self.frame_count > 3000:
            dead = True
            reward = -3 if not self.ppo_optimized else -1

        info = {"errors": "No errors"}  # Additional info for debugging, if necessary

        # Modify the return statement based on mode
        frame_skip_frames = np.zeros((self.frame_stack_num, 64, 96), dtype=np.float32)

        for i in range(self.frame_stack_num):
            index = -1 - (self.frame_skip * i)
            frame_skip_frames[self.frame_stack_num - 1 - i] = self.frame_stack[index]

        if self.ppo_optimized:
            # PPO mode - return simplified observation
            return frame_skip_frames, reward, dead, False, info
        else:
            # DQN mode - return dictionary observation
            observation = {
                'frames': frame_skip_frames,
                'action_history': np.array(self.action_history, dtype=np.int32)
            }
            return observation, reward, dead, False, info

    def reset(self, save_movie=False, episode=None):
        """
        Reset the state of the environment to an initial state
        """
        # Save the video
        if save_movie:
            create_episode_video(self.episode_frames, episode)
        
        self.episode_frames = []
        self.frame_count = 0
        self.step_count = 0

        # Reset position tracking for PPO
        self.prev_x_pos = 0
        self.max_x_pos = 0

        # Reset action history (only for DQN mode)
        if not self.ppo_optimized:
            self.action_history = [0] * self.action_history_length

        # Load the savestate
        self.saver.load_file('W1-3.sav')

        self.frame_stack = np.zeros(((self.frame_skip * self.frame_stack_num), 64, 96), dtype=np.float32)

        self.state = None 

        # Return observation based on mode
        if self.ppo_optimized:
            # PPO mode - return simplified observation
            return np.zeros((self.frame_stack_num, 64, 96), dtype=np.float32)
        else:
            # DQN mode - return dictionary observation
            initial_observation = {
                'frames': np.zeros((self.frame_stack_num, 64, 96), dtype=np.float32),
                'action_history': np.array(self.action_history, dtype=np.int32)
            }
            return initial_observation

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.emu.screenshot()
        elif mode == 'human':
            self.window.draw()

    def close(self):
        self.emu.destroy()
        self.window.destroy()
        
    def _calculate_ppo_reward(self, dead):
        """
        Calculate a more informative reward for PPO training
        """
        # Get Mario's current position
        current_x_pos = self.emu.memory.signed[0x021B6A90:0x021B6A90:4]
        
        # Progress reward - encourage moving right
        progress_reward = (current_x_pos - self.prev_x_pos) / 1000.0
        
        # Max progress bonus - reward for reaching new maximum positions
        max_progress_bonus = 0
        if current_x_pos > self.max_x_pos:
            max_progress_bonus = 0.5
            self.max_x_pos = current_x_pos
        
        # Time penalty - small penalty to encourage faster completion
        time_penalty = -0.01
        
        # Inactivity penalty - penalize staying in same position
        inactivity_penalty = 0
        if abs(current_x_pos - self.prev_x_pos) < 50:  # Very small movement
            inactivity_penalty = -0.02
        
        # Death penalty
        death_penalty = -10 if dead else 0
        
        # Update previous position
        self.prev_x_pos = current_x_pos
        
        # Combine all rewards
        total_reward = progress_reward + max_progress_bonus + time_penalty + inactivity_penalty + death_penalty
        
        return total_reward

