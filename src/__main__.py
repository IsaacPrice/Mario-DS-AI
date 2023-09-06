from GameRunning import GameLoop
from Window import Window
import numpy as np
import multiprocessing

np_stupid = [0, 0, 0, 0, 0, 0, 0, 0]

data = {
    'model_data' : {
        'running' : 1, # -1 means that it is stopped, 0 means that it is paused, and 1 is currently running
        'save_every' : 108000 # This is every half hour if going at 60fps
    },
    'game_data' : {
        'actions' : np.array(np_stupid),
        'velocity' : 0,
        'reward' : 0,
        'save?' : 0,
        'reward_calc' : {
            'coins' : .2,
            'movement' : .3,
            'power-up' : .2
        }
    }
}

def run_ai():
    game = GameLoop()
    window = Window()
    total_frames = 0

    # Keep on updating the game
    while True:
        temp_data = game.cycle(data['game_data'])

        # Check if the user wanted to close the game
        if temp_data is None:
            pass # TODO: Make this close everything without any errors
        else: 
            data['game_data'] = temp_data

        total_frames += 1
        if total_frames % data['model_data']['save_every'] == 0:
            data['game_data']['save?'] = 1
        window.update_labels(data['game_data']['actions'])


run_ai()
