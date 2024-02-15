import gym
from gym import spaces
import numpy as np
from Input import Input
from DataProccesing import preprocess_image_numpy
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor

class MarioDSEnv(gym.Env):
    """
    Custom Environment for Mario DS that follows gym interface.
    This class will be a wrapper around the Nintendo DS emulator to play Mario DS,
    handling state representation, actions, and rewards.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(MarioDSEnv, self).__init__()

        # Create action and observation space
        self.action_space = spaces.Discrete(8) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 96, 128), dtype=np.float16)

        # Initialize the emulator and extras
        self.emu = DeSmuME()  
        self.emu.open('NSMB.nds')
        self.window = self.emu.create_sdl_window()
        self.saver = DeSmuME_Savestate(self.emu)
        self.saver.load_file('W1-3.sav')
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
        self.frame_stack = np.zeros((4, 96, 128), dtype=np.float16)

    def step(self, action):
        """
        Execute one time step within the environment
        """
        # Update the current Framestack
        self.current_stack = self.frame_stack

        # 1. Send the action to the emulator
        self.action_mapping[action]()

        # Move the emulator forward by 1 frame
        self.emu.cycle()

        # 2. Obtain the next state from the emulator
        frame = self.emu.screenshot()
        self.frame_array, dead = preprocess_image_numpy(frame)
        self.frame_stack = np.concatenate((self.frame_stack, self.frame_array.reshape(1, 96, 128)))
        self.frame_stack = self.frame_stack[1:, :, :]  # Remove the oldest frame
        print(self.frame_stack.shape)

        # 3. Calculate the reward
        reward = (self.emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000) - 0.02
        if dead:
            reward = -1

        info = {"errors": "No errors"}  # Additional info for debugging, if necessary
        return np.array(self.frame_stack), reward, dead, False, info  # Return four values instead of five

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """

        # Reset the emulator to the start of the game/level
        self.saver.load_file('W1-3.sav')
        self.frame_stack = np.zeros((4, 96, 128), dtype=np.float16)

        self.state = None 
        return np.array(self.frame_stack)  # Return the initial state

    def render(self, mode='human'):
        """
        Render the environment to the screen
        """
        if mode == 'rgb_array':
            return self.emu.screenshot()
        elif mode == 'human':
            self.window.draw()

    def close(self):
        """
        Perform any necessary cleanup
        """
        pass # No cleanup needed
