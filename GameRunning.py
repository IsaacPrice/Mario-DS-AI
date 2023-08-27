from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from desmume.controls import Keys, load_configured_config, keymask
import numpy as np
from collections import deque
from DataProccesing import preprocess_image
from AI import MarioDQN

# These are all the inputs that we need the AI to use
keys = [Keys.KEY_A, Keys.KEY_X, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_DOWN, Keys.KEY_UP]

def release_all():
    for key in keys:
        emu.input.keypad_rm_key(keymask(key))

def none():
    release_all()

def jump():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_A))

def jump_left():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_A))
    emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))

def jump_right():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_A))
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

def walk_left():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))

def walk_right():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

def run_left():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_X))
    emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))

def run_right():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_X))
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

def down():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_DOWN))

def up():
    release_all()
    emu.input.keypad_add_key(keymask(Keys.KEY_UP))

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
frame_stack = np.zeros(35280)

# Initialize your AI
state_shape = 7056 * 5  # This should be 35280
n_actions = 9  # The number of actions your AI can take
mario_agent = MarioDQN(state_shape, n_actions)

# Action mapping
action_mapping = {
    0: none,
    1: jump,
    2: jump_left,
    3: jump_right,
    4: walk_left,
    5: walk_right,
    6: run_left,
    7: run_right,
    8: down,
    9: up,
}

# Run the emulation as fast as possible until quit
while not window.has_quit():
    window.process_input()   # Controls are the default DeSmuME controls, see below.
    
    # Get the current state (stacked frames)
    current_state = frame_stack  # This is your state representation

    # Choose an action
    action = mario_agent.choose_action(current_state)

    # Perform the action
    action_mapping[action]()

    emu.cycle()
    window.draw()

    # Update the current previous 5 frames
    frame = emu.screenshot()
    processed_frame = preprocess_image(frame)
    frame_stack = frame_stack[7056:]
    frame_stack = np.append(frame_stack, processed_frame)


    # Gets the amount of curennt lives
    mem_acc = MemoryAccessor(False, emu.memory)
    lives = mem_acc.read_byte(0x2208b364)
    if lives < 4:
        print('Died, restarting...')
        saver.load_file('W1-1.sav')

    # Calculate the reward
    # 12288 is the max speed, generally. we will make it less just in case the AI gets too fast
    leftMovement = mem_acc.read_long(0x021B6A90) / 20000

    reward = leftMovement / 3

    mario_agent.learn(current_state, action, reward, frame_stack)
    
    # Print the movement over the other movement line
    print('Movement: ' + str(leftMovement) + ' Lives: ' + str(lives), end='\r')