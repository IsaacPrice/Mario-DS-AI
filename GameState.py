import random

memory_addresses = {
    "mario horizonal speed": 0x0057,
    "mario position x": 0x006D,
    "lives": 0x075A, # To tell if they died at some point
    "time": 0x07F8,
    "coins": 0x075E,
    "Powerup": 0x0756
}

# This class will be used to find certain data in the game, like if mario is alive or dead, and for giving the reward
class GameState:
    def __init__(self):
        pass

    def CalculateReward(self):
        # Firstly we are going to get all of the values we need
        pass

    # This will run when mario dies, will reset the game and start the learning process
    def whenDead(self):
        # Reset the game to a random level
        randomLevel = random.randint(0, 10)

        # Load the save state

    # this will run every frame
    def tick(self):
        reward = self.CalculateReward()