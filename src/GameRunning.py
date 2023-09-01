from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
import numpy as np
import keyboard
from DataProccesing import preprocess_image
from AI import MarioDQN
from Input import Input
from DebugInput import DebugInput
import json 
import time
from PyQt5.QtWidgets import *
from flask import Flask, jsonify, request

data = {
    'model_data' : {
        'layers' : [
            ['Dense', 128],
            ['Dense', 24],
            ['Dense', 24],
            ['Dense', 10]
        ],
        'alpha' : 0,
        'gamma' : 0,
        'epsilon' : 0
    },
    'game_data' : {
        'velocity' : 0,
        'reward' : 0
    },
    'running' : 1 # -1 means that it is stopped, 0 means that it is paused, and 1 is currently running
}

def game_AI():
    path = "C:/Programs/Mario-DS-AI/"

    app = Flask(__name__)

    @app.route('/get_data', methods=['GET'])
    def get_data():
        return jsonify(data)

    @app.route('/update_data', methods=['POST'])
    def update_data():
        key = request.json.get('key')
        value = request.json.get('value')
        data[key] = value
        return jsonify({"message": "Updated successfully"})
    
    app.run()

    # Creating the emulator & opening files
    emu = DeSmuME()
    emu.open(path + 'NSMB.nds')
    window = emu.create_sdl_window()
    saver = DeSmuME_Savestate(emu)
    saver.load_file(path + 'save_files/W1-1.sav')
    mem = DeSmuME_Memory(emu)


    # DATA FOR THE AI
    frames = [] # This will be the list of frames that will be used as the input for the AI
    reward = 0

    # Load the config file
    with open(path + 'settings/config.json', 'r') as f:
        config_data = json.load(f)

    # Get the neccisary data from the file
    AMOUNT_OF_FRAMES = config_data['ModelSettings']['FrameStackAmount']
    update_every = config_data['ModelSettings']['UpdateEveryNFrame']

    # Get the input shape and base values 
    TOTAL_PIXELS = AMOUNT_OF_FRAMES * 7056
    frame_stack = np.zeros(TOTAL_PIXELS)
    n_actions = 9  # The number of actions the AI can take

    # Creat the AI
    mario_agent = MarioDQN(TOTAL_PIXELS, n_actions, TOTAL_PIXELS)

    # Make a class object for the input
    inputs = Input(emu)
    key_inputs = DebugInput(inputs, config_data['Inputs'])

    # Action mapping
    action_mapping = {
        0: inputs.none,
        1: inputs.jump,
        2: inputs.jump_left,
        3: inputs.jump_right,
        4: inputs.walk_left,
        5: inputs.walk_right,
        6: inputs.run_left,
        7: inputs.run_right,
        8: inputs.down,
        9: inputs.up
    }

    total_reward = 0
    amount = 0


    # Run the emulation as fast as possible until quit
    while not window.has_quit():
        window.process_input() # Controls are the default DeSmuME controls, which are always wrong
        
        # Get the current state (stacked frames)
        current_state = frame_stack 

        # Choose an action
        user_action = key_inputs.PollKeyboard(inputs)

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

        data['game_data']['velocity'] = Movement

        if amount % update_every == 0:
            mario_agent.learn(current_state, action, total_reward, frame_stack)
            data['game_data']['reward'] = total_reward
            total_reward = 0
        
        # Checks the current state of the emulation
        if data['running'] == 0:
            while data['running'] != 0:
                time.sleep(.33)
        elif data['running'] == -1:
            pass # TODO: Make this exit the application without saving it it has been closed

game_AI()