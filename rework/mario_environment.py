import gym 
import numpy as np

from typing import Tuple
from collections import deque
from gym import spaces
from desmume.emulator import DeSmuME, DeSmuME_Savestate
from Input import Input
import torch



import torch.nn.functional as F

def ProcessImage(image, width_reduction=4, height_reduction=4) -> Tuple[torch.Tensor, bool]:
    """
    Preprocess the image for the Mario DS environment.
    This function by itself will cut down the Fps by about 15 FPS

    Returns:
    Tuple[torch.Tensor, bool]: The preprocessed image and a boolean indicating if the player is dead.
    """

    # Convert the image to torch tensor and move it to GPU
    converted_image = torch.tensor(np.array(image.convert('L'))).cuda()
    cropped_image = converted_image[0:192, 0:256]
    resized_image = cropped_image[::height_reduction, ::width_reduction]

    # Check for the death condition
    line_sum = torch.sum(converted_image[237, :])

    return resized_image.float() / 255, line_sum == 44044


class MarioDSEnv(gym.Env):
    """
    Custom Environment for Mario DS that follows gym interface.
    This class will be a wrapper around the Nintendo DS emulator to play Mario DS,
    handling state representation, actions, and rewards.
    """

    def __init__(self, frame_skip: int, observation_frames: int, step_limit: int = 3000, rom_path: str = 'NSMB.nds', muted: bool = True):
        """
        Initialize the Mario DS environment.

        Args:
        frame_skip (int): The number of frames that seperate each frame in the stack.
        frame_stack (int): The number of frames to stack together.
        rom_path (str, optional): The path to the ROM file. Defaults to 'NSMB.nds'.
        savestate_path (str, optional): The path to the savestate file. Defaults to 'W1-1.sav'.
        muted (bool, optional): Whether to mute the emulator. Defaults to True.
        """
        super(MarioDSEnv, self).__init__()

        # Create action and observation space
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=1, shape=(observation_frames, 48, 64), dtype=np.float32)
        self.frame_skip = frame_skip
        self.num_frames = observation_frames
        self.step_limit = step_limit
        self.current_step = 0 # This is used to timeout the episode

        # Configure the emulator
        self.emu = DeSmuME()
        self.emu.open(rom_path)
        self.window = self.emu.create_sdl_window()
        self.saver = DeSmuME_Savestate(self.emu)
        self.saver.load_file('W1-1 (linux).dsv')
        self.emu.volume_set(0 if muted else 100)
        self.inputs = Input(self.emu)
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
        self.frame_stack = deque(maxlen=observation_frames * frame_skip)
        self.reset()

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
        action (int): The action to take.

        Returns:
        Tuple[np.ndarray, float, bool, bool, dict]: The next state, the reward, whether the episode is done, False (Gym requires it) and additional information.
        """
        """
        Reset the environment.

        Returns:
        np.ndarray: The initial state of the environment.
        """
        # Apply the action
        self.action_mapping[action]()
        self.emu.cycle()

        # Get the next frame, process it, and update the frame stack
        frame = self.emu.screenshot()
        frame, dead = ProcessImage(frame)
        self.frame_stack.append(frame)
        self.current_step += 1

        # Calculate the reward and done
        if self.current_step >= self.step_limit:
            dead = True
        reward = (self.emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000) - 0.02
        if dead:
            reward = -1

        # Later we will add a diagnostics class to store additional information, and thats where info will go
        # Will NOT include the frames, that will be accessable through the render method
        # TODO: Add diagnostics class
        info = {'Diagnostic': False}

        # Get the proper state
        selected_frames = list(self.frame_stack)[::self.frame_skip]
        state = torch.stack(selected_frames).cuda()

        # Return the state, reward, done, False (Gym requires this), and info
        return state, reward, dead, False, info

    def reset(self):

        # Reset the emulator and the frame stack
        self.saver.load_file('W1-1 (linux).dsv')
        self.current_step = 0

        # Fill the frame stack with the initial frame
        frame = self.emu.screenshot()
        frame, _ = ProcessImage(frame)
        for _ in range(self.num_frames * self.frame_skip):
            self.frame_stack.append(frame)

        # Get the proper state
        selected_frames = list(self.frame_stack)[::self.frame_skip]

        state = torch.stack(selected_frames).cuda()

        return state

    def render(self, mode='human'):

        if mode == 'human':
            self.window.draw()
        elif mode == 'rgb_array':
            return self.emu.screenshot()
    
    def close(self):
        self.emu.destroy()
        self.window.destroy()
        



        



        