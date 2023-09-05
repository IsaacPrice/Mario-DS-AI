from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
import numpy as np
from DataProccesing import preprocess_image
from AI import MarioDQN
from Input import Input
from DebugInput import DebugInput
import json 
from PyQt5.QtWidgets import *

class GameLoop:
    def __init__(self, filepath='C:/Programs/Mario-DS-AI/'):
        self.filepath = filepath

        # Create the emulator, game window, and saver
        self.emu = DeSmuME()
        self.emu.open(self.filepath + 'NSMB.nds')
        self.window = self.emu.create_sdl_window()
        self.saver = DeSmuME_Savestate(self.emu)
        
        # Load the game
        self.saver.load_file(self.filepath + 'save_files/W1-1.sav')

        # Load the config from JSON file
        with open(self.filepath + 'settings/config.json', 'r') as f:
            self.config_data = json.load(f)

        # Create the data for the AI
        self.reward = 0
        self.total_reward = 0
        self.amount = 0

        self.AMOUNT_OF_FRAMES = self.config_data['ModelSettings']['FrameStackAmount']
        self.UPDATE_EVERY = self.config_data['ModelSettings']['UpdateEveryNFrame']

        self.TOTAL_PIXELS = self.AMOUNT_OF_FRAMES * 7056
        self.frame_stack = np.zeros(self.TOTAL_PIXELS)
        self.n_actions = 8

        # Create the AI
        self.mario_agent = MarioDQN(self.TOTAL_PIXELS, self.n_actions, self.TOTAL_PIXELS)

        # Create inputs
        self.inputs = Input(self.emu)
        self.key_inputs = DebugInput(self.inputs, self.config_data['Inputs'])

        # Create the action mapping
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
    
    def cycle(self, game_data):
        if self.window.has_quit(): 
            return None # The game exited, meaning we need to shut down the rest of the stuff safely

        # This will get certain inputs from the window
        self.window.process_input()

        self.current_state = self.frame_stack # Create a previous state for learning
        
        # Deal with inputs
        action = self.key_inputs.poll_keyboard(self.inputs)
        self.confidence = self.mario_agent.choose_action(self.current_state) # This is how much the AI wants to take each action
        if action == 0:
            action = np.argmax(self.confidence)
        self.action_mapping[action]()

        # Update the data
        game_data['actions'] = self.confidence

        # Move the emulator along
        self.emu.cycle()
        self.window.draw()

        # Update the frame data
        frame = self.emu.screenshot()
        self.processed_frame, dead = preprocess_image(frame)
        self.frame_stack = self.frame_stack[7056:]
        self.frame_stack = np.append(self.frame_stack, self.processed_frame)

        # Punishes the AI when mario dies and restarts the level
        if dead: 
            self.total_reward -= 3
            self.saver.load_file(self.filepath + 'save_files/W1-1.sav')

        # Get the reward
        self.movement = self.emu.memory.signed[0x021B6A90:0x021B6A90:4]
        self.reward = self.movement / 20000
        self.reward -= .1

        self.total_reward += self.reward
        self.amount += 1

        # Update the AI when neccisary
        if self.amount % self.UPDATE_EVERY == 0:
            self.mario_agent.learn(self.current_state, action, self.total_reward, self.frame_stack)
            self.total_reward = 0
            self.amount = 0
        
        if game_data['save?'] == 1:
            self.mario_agent.save()

        return game_data            


'''def game_AI():
    path = "C:/Programs/Mario-DS-AI/"
    
    # Creating the emulator & opening files
    emu = DeSmuME()
    emu.open(path + 'NSMB.nds')
    window = emu.create_sdl_window()
    saver = DeSmuME_Savestate(emu)
    saver.load_file(path + 'save_files/W1-1.sav')

    # DATA FOR THE AI
    reward = 0
    total_reward = 0
    amount = 0

    # Load the config file
    with open(path + 'settings/config.json', 'r') as f:
        config_data = json.load(f)

    # Get the neccisary data from the file
    AMOUNT_OF_FRAMES = config_data['ModelSettings']['FrameStackAmount']
    update_every = config_data['ModelSettings']['UpdateEveryNFrame']

    # Get the input shape and base values 
    TOTAL_PIXELS = AMOUNT_OF_FRAMES * 7056
    frame_stack = np.zeros(TOTAL_PIXELS)
    n_actions = 7  # The number of actions the AI can take

    # Creat the AI
    mario_agent = MarioDQN(TOTAL_PIXELS, n_actions, TOTAL_PIXELS)

    # Make a class object for the input
    inputs = Input(emu)
    key_inputs = DebugInput(inputs, config_data['Inputs'])

    # Action mapping
    action_mapping = {
        0: inputs.none,
        1: inputs.walk_left,
        2: inputs.walk_right,
        3: inputs.run_left,
        4: inputs.run_right,
        5: inputs.jump,
        6: inputs.jump_left,
        7: inputs.jump_right
    }


    # Run the emulation as fast as possible until quit
    while not window.has_quit():
        window.process_input() # Controls are the default DeSmuME controls, which are always wrong
        
        # Get the current state (stacked frames)
        current_state = frame_stack 

        # Choose an action
        user_action = key_inputs.poll_keyboard(inputs)

        if user_action > 0:
            action_mapping[user_action]()
        else:
            action = mario_agent.choose_action(current_state)
            action_mapping[action]()
        # Gets the inputs from the user if any

        emu.cycle()
        window.draw()

        # Update the current previous 5 frames
        frame = emu.screenshot()
        processed_frame, dead = preprocess_image(frame)
        frame_stack = frame_stack[7056:]
        frame_stack = np.append(frame_stack, processed_frame)

        if dead == True:
            print('Died, restarting...')
            total_reward -= 3
            saver.load_file(path + 'save_files/W1-1.sav')

        # Calculate the reward
        # 12288 is the max speed, generally. we will make it less just in case the AI gets too fast
        Movement = emu.memory.signed[0x021B6A90:0x021B6A90:4]

        reward = Movement / 20000 # Get the reward just from the movement
        reward = reward - 0.1 # Punish the AI for not moving

        total_reward += reward
        amount += 1

        if amount % update_every == 0:
            mario_agent.learn(current_state, action, total_reward, frame_stack)
            total_reward = 0

game_AI()'''