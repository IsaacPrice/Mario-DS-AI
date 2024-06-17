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

emu = DeSmuME()
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)

class MarioDSEnv(gym.Env):
    """
    Custom Environment for Mario DS that follows gym interface.
    This class will be a wrapper around the Nintendo DS emulator to play Mario DS,
    handling state representation, actions, and rewards.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, frame_skip, frame_stack):
        global emu, window, saver
        super(MarioDSEnv, self).__init__()

        # Create action and observation space
        self.action_space = spaces.Discrete(8) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 48, 64), dtype=np.float16)
        self.frame_skip = frame_skip
        self.frame_stack_num = frame_stack

        # Initialize the emulator and extras
        # self.emu = DeSmuME()
        # self.emu.open('NSMB.nds')
        # self.window = self.emu.create_sdl_window()
        # self.saver = DeSmuME_Savestate(self.emu)
        # Load the savestate, random from 1 or 4
        saver.load_file('W1-1.sav')

        self.frame_count = 0
        self.inputs = Input(emu)
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
        # Create the empty frame stack
        emu.volume_set(0)
        self.frame_stack = np.zeros(((self.frame_skip * self.frame_stack_num), 48, 64), dtype=np.float16)
        self.episode_frames = []
        self.reset()

    def step(self, action):
        """
        Execute one time step within the environment
        """
        global emu, window, saver
        # Update the current Framestack
        self.current_stack = self.frame_stack

        # 1. Send the action to the emulator
        self.action_mapping[action]()

        # Move the emulator forward by 1 frame
        emu.cycle()
        self.frame_count += 1

        # 2. Obtain the next state from the emulator
        frame = emu.screenshot()
        self.episode_frames.append(frame)
        dead = False
        self.frame_array, dead = preprocess_image_numpy(frame)
        self.frame_stack = np.concatenate((self.frame_stack, self.frame_array.reshape(1, 48, 64))) 
        self.frame_stack = self.frame_stack[1:, :, :]  # Remove the oldest frame

        # 3. Calculate the reward
        reward = (emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000) - 0.02
        if dead or self.frame_count > 3000:
            dead = True
            reward = -3

        info = {"errors": "No errors"}  # Additional info for debugging, if necessary

        # Modify the return statement
        frame_skip_frames = np.zeros((self.frame_stack_num, 48, 64), dtype=np.float16)

        for i in range(self.frame_stack_num):
            index = -1 - (self.frame_skip * i)
            frame_skip_frames[self.frame_stack_num - 1 - i] = self.frame_stack[index]

        return frame_skip_frames, reward, info, dead, False

    def reset(self, save_movie=False, episode=None):
        """
        Reset the state of the environment to an initial state
        """
        global emu, window, saver
        # Save the video
        if save_movie:
            create_episode_video(self.episode_frames, episode)
        
        self.episode_frames = []
        self.frame_count = 0

        # Load the savestate

        # Load the savestate, random from 1 or 4
        saver.load_file('W1-1.sav')

        self.frame_stack = np.zeros(((self.frame_skip * self.frame_stack_num), 48, 64), dtype=np.float16)

        self.state = None 

        return np.zeros((self.frame_stack_num, 48, 64), dtype=np.float16)

    def render(self, mode='human'):
        """
        Render the environment to the screen
        """
        global emu, window, saver
        if mode == 'rgb_array':
            return emu.screenshot()
        elif mode == 'human':
            window.draw()

    def close(self):
        """
        Perform any necessary cleanup
        """

