from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
import numpy as np
import keyboard
from DataProccesing import preprocess_image
from AI import MarioDQN
from Input import Input

emu = DeSmuME()
emu.open('NSMB.nds')

# Create the window for the emulator
window = emu.create_sdl_window()

# DATA FOR THE AI
frames = [] # This will be the list of frames that will be used as the input for the AI
reward = 0

saver = DeSmuME_Savestate(emu)
saver.load_file('W1-1.sav')

mem = DeSmuME_Memory(emu)

# Initialize a deque with maxlen 5 filled with zeros
AMOUNT_OF_FRAMES = 1
TOTAL_PIXELS = AMOUNT_OF_FRAMES * 7056
frame_stack = np.zeros(TOTAL_PIXELS)

# Initialize your AI
state_shape = TOTAL_PIXELS # This should be 21168
n_actions = 9  # The number of actions your AI can take
mario_agent = MarioDQN(state_shape, n_actions, TOTAL_PIXELS)

# Make a class object for the input
inputs = Input(emu)

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
    current_state = frame_stack  # This is your state representation

    # Choose an action
    action = mario_agent.choose_action(current_state)

    # Perform the action
    action_mapping[action]()

    # Check for keyboard inputs, and moves if so
    try:
        if keyboard.is_pressed('x'):
            inputs.release_all()
            inputs.jump()
        elif keyboard.is_pressed('j'):
            inputs.release_all()
            inputs.run_left()
        elif keyboard.is_pressed('l'):
            inputs.release_all()
            inputs.run_right()
        '''elif keyboard.is_pressed('a'):
            saver.save_file('W1-1.sav')'''
    except:
        pass

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
        saver.load_file('W1-1.sav')

    # Calculate the reward
    # 12288 is the max speed, generally. we will make it less just in case the AI gets too fast
    Movement = emu.memory.signed[0x021B6A90:0x021B6A90:4]

    reward = Movement / 20000 # Get the reward just from the movement
    reward = reward - 0.1 # Punish the AI for not moving

    total_reward += reward
    amount += 1

    if amount % 2 == 0:
        mario_agent.learn(current_state, action, total_reward, frame_stack)
        print(total_reward, end='\r')
        total_reward = 0