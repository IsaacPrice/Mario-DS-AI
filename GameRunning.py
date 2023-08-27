from desmume.emulator import DeSmuME
from desmume.controls import keymask, Keys
from PIL import ImageGrab
import time

emu = DeSmuME()
emu.open('NSMB.nds')

# Create the window for the emulator
window = emu.create_sdl_window()

memory_addresses = {
    "mario horizonal speed": 0x0057,
    "mario position x": 0x006D,
    "lives": 0x075A, # To tell if they died at some point
    "time": 0x07F8,
    "coins": 0x075E,
    "Powerup": 0x0756
}

# Load savestate World 1-1
# emu.load_state('NSMB.sav')

# DATA FOR THE AI
frames = [] # This will be the list of frames that will be used as the input for the AI
reward = 0

# Run the emulation as fast as possible until quit
while not window.has_quit():
    # This will take the last frame after being calculated through the AI and run the controls 

    window.process_input()   # Controls are the default DeSmuME controls, see below.
    emu.cycle()
    window.draw()

    # Update the current previous 5 frames
    # img = ImageGrab.grab(bbox=(x1, y1, x2, y2))  # x1, y1, x2, y2 are the coordinates of the bounding box

    # Determine if mario has died or not by getting the lives value compared to the last frame
    #current_lives = emu.read_memory(memory_addresses["lives"])

    # Calculate the reward

