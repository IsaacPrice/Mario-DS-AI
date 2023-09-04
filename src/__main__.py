from GameRunning import GameLoop
from Window import Window
import numpy as np
import multiprocessing

np_stupid = [0, 0, 0, 0, 0, 0, 0, 0]

data = {
    'model_data' : {
        'running' : 1, # -1 means that it is stopped, 0 means that it is paused, and 1 is currently running
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
        'actions' : np.array(np_stupid),
        'velocity' : 0,
        'reward' : 0
    }
}

def run_ai():
    game = GameLoop()
    window = Window()

    # Keep on updating the game
    while True:
        temp_data = game.cycle(data['game_data'])

        # Check if the user wanted to close the game
        if temp_data is None:
            pass # TODO: Make this close everything without any errors
        else: 
            data['game_data'] = temp_data
        
        window.update_labels(data['game_data']['actions'])

run_ai()

'''if __name__ == '__main__':
    multiprocessing.freeze_support()  # Optional, only if you're going to freeze your script into an executable
    
    ai_process = multiprocessing.Process(target=run_ai)
    window_process = multiprocessing.Process(target=run_window)

    ai_process.start()
    window_process.start()

    ai_process.join()
    window_process.join()'''