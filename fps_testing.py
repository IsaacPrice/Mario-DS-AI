from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from desmume.controls import Keys, keymask
import numpy as np
import time



emu = DeSmuME()  
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
emu.reset()
saver.load_file('W1-1 (linux).dsv')
emu.volume_set(100)

import keyboard
from desmume.controls import Keys, keymask

keys = [Keys.KEY_A, Keys.KEY_X, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_DOWN, Keys.KEY_UP]
def release_all():
    for key in keys:
        emu.input.keypad_rm_key(keymask(key))

        
frames = 0
start_time = time.time()

"""
Notes:
 - The emulator, uncapped without modifications, runs at around 125 FPS
 - The Screenshot takes it down to about 120 FPS
 - Converting the image to grayscale takes it down to about 118 FPS
 - If all doing the conversions with numpy, the frames are around 110 FPS

"""

while (True):
    window.process_input()
    image = emu.screenshot()
    converted_image = np.array(image.convert('L'))
    cropped_image = converted_image[0:192, 0:256]
    resized_image = cropped_image[::4, ::4]
    line_sum = np.sum(converted_image[237, :])

    """
    line_sum = np.sum(np.array(image.convert('L').crop((0, 237, 256, 238))))
    if line_sum == 44044:
        saver.load_file('W1-1 (linux).dsv')
    """
    emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    emu.cycle()
    window.draw()

    frames += 1
    current_time = time.time()
    if current_time - start_time >= 1.0:
        print(f"FPS: {frames}")
        frames = 0
        start_time = current_time



emu.destroy() 
window.destroy()
