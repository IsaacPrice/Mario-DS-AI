import pyautogui

keys = ['up', 'left', 'right', 'down', 'a', 's']

# This class will be used to control mario
class Input:
    def __init__(self):
        pass

    # This will release all keys
    def releaseAll(self):
        for key in keys:
            pyautogui.keyUp(key)
    
    # This will make mario do nothing
    def none(self):
        self.releaseAll()

    # This will make mario jump
    def jump(self):
        self.releaseAll()
        pyautogui.keyDown('s')

    # This will make mario walk left
    def left(self):
        self.releaseAll()
        pyautogui.keyDown('left')

    # This will make mario run left
    def runLeft(self):
        self.releaseAll()
        pyautogui.keyDown('left')
        pyautogui.keyDown('a')

    # This will make mario walk right
    def right(self):
        self.releaseAll()
        pyautogui.keyDown('right')

    # This will make mario run right
    def runRight(self):
        self.releaseAll()
        pyautogui.keyDown('right')
        pyautogui.keyDown('a')

    # This will make mario down (crouch if not in the air, or ground pound if in the air)
    def down(self):
        self.releaseAll()
        pyautogui.keyDown('down')

    # This will press on the screen where the button for the powerup normally is
    def powerUpTap(self):
        pass
