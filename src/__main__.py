from GameRunning import game_AI
from Window import main_window
import multiprocessing

data = {
    'model_data' : {
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
        'velocity' : 0,
        'reward' : 0
    },
    'running' : 1 # -1 means that it is stopped, 0 means that it is paused, and 1 is currently running
}

def run_ai():
    game_AI(data)

def run_window():
    main_window(data)


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Optional, only if you're going to freeze your script into an executable
    
    ai_process = multiprocessing.Process(target=run_ai)
    window_process = multiprocessing.Process(target=run_window)

    ai_process.start()
    window_process.start()

    ai_process.join()
    window_process.join()