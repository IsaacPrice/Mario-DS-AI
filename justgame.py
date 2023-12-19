import time
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor

emu = DeSmuME()
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
saver.load_file('W1-1.sav')

frame_count = 0
start_time = time.time()

while True:
    window.process_input()
    emu.cycle()
    window.draw()

    # Calculate the frame rate
    frame_count += 1
    if frame_count % 60 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 60 / elapsed_time
        print(f"Frame Rate: {fps:.1f}")
        start_time = time.time()
