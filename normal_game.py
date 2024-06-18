from DataProccesing import preprocess_image_numpy
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from DataProccesing import preprocess_image_numpy

emu = DeSmuME()  
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
saver.load_file('W1-5.sav')
emu.volume_set(100)

import keyboard
from desmume.controls import Keys, keymask

class Input:
    def __init__(self, emu):
        self.emu = emu
        self.keys = {
            'a': Keys.KEY_A,
            's': Keys.KEY_B,
            'd': Keys.KEY_X,
            'f': Keys.KEY_Y,
            'q': Keys.KEY_L,
            'e': Keys.KEY_R,
            'enter': Keys.KEY_START,
            'space': Keys.KEY_SELECT,
            'up': Keys.KEY_UP,
            'down': Keys.KEY_DOWN,
            'left': Keys.KEY_LEFT,
            'right': Keys.KEY_RIGHT
        }

    def update(self):
        for key, ds_key in self.keys.items():
            if keyboard.is_pressed(key):
                self.emu.input.keypad_add_key(keymask(ds_key))
            else:
                self.emu.input.keypad_rm_key(keymask(ds_key))

inputs = Input(emu)


while (True):
    inputs.update()
    emu.cycle()
    window.draw()

    if keyboard.is_pressed('esc'):
        break

    #frame = emu.screenshot()
    #frame, dead = preprocess_image_numpy(frame)
emu.destroy() 
window.destroy()

while (True):
    pass 

#emu.close()

