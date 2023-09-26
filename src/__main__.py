from GameRunning import GameLoop
from console import Dashboard
import numpy as np

data = {
    'model_data' : {
        'running' : 1, # -1 means that it is stopped, 0 means that it is paused, and 1 is currently running
        'save_every' : 108000 # This is every half hour if going at 60fps
    },
    'game_data' : {
        'actions' : np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        'q_values' : np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        'velocity' : 0,
        'reward' : 0,
        'total_reward' : 0,
        'save?' : 0,
        'level' : 'W1-1.sav',
        'died' : 0,
        'reward_calc' : {
            'coins' : .5,
            'movement' : .7,
            'power-up' : 1,
            'points-scale' : .01 # A coin is 100, killng an enemy is 200 * streak, powerup is ---, and there are other important 
        }
    }
}

def run_ai():
    game = GameLoop('C:/Users/jpric/OneDrive/Desktop/poopoo/Mario-DS-AI/')
    dashboard = Dashboard()
    total_frames = 0

    # Keep on updating the game
    while True:
        temp_data = game.cycle(data['game_data'])
        died = data['game_data']['died']

        # Check if the user wanted to close the game
        if temp_data is None:
            pass # TODO: Make this close everything without any errors
        else: 
            data['game_data'] = temp_data

        # This will save the model whenever it has been a bit of time
        total_frames += 1
        if total_frames % data['model_data']['save_every'] == 0:
            data['game_data']['save?'] = 1

        # Update the console window
        dashboard.update(data['game_data'])

run_ai()
