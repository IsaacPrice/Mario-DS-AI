from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from desmume.controls import Keys, load_configured_config, keymask
import keyboard

emu = DeSmuME()
emu.open('NSMB.nds')

# Create the window for the emulator
window = emu.create_sdl_window()

saver = DeSmuME_Savestate(emu)
#saver.load_file('W1-1.sav')

while not window.has_quit():
    window.process_input()

    if keyboard.is_pressed('l'):
        emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))
    elif keyboard.is_pressed('j'):
        saver.save_file('W1-1.sav')
    else:
        emu.input.keypad_rm_key(keymask(Keys.KEY_RIGHT))

    emu.cycle()
    window.draw()