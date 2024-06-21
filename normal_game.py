from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from desmume.controls import Keys, keymask
import numpy as np
from PIL import Image


emu = DeSmuME()  
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
emu.reset()
#saver.load_file('poop.dsv')
emu.volume_set(100)

import keyboard
from desmume.controls import Keys, keymask

keys = [Keys.KEY_A, Keys.KEY_X, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_DOWN, Keys.KEY_UP]
def release_all():
    for key in keys:
        emu.input.keypad_rm_key(keymask(key))
        
frame = 0
while (True):
    window.process_input()
    image = emu.screenshot()
    line_sum = np.sum(np.array(image.convert('L').crop((0, 237, 256, 238))))
    if line_sum == 43342:
        frame += 1
    if frame == 30:
        emu.pause() 
        saver.save_file('W1-1 (linux).dsv')
    else:
        emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    emu.cycle()
    window.draw()


emu.destroy() 
window.destroy()

while (True):
    pass 

#emu.close()

