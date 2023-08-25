import pyautogui

keys = ['up', 'left', 'right', 'down', 'a', 's']

class Input:
    def __init__(self):
        pass
    
    # This will make mario do nothing
    def none(self):
        pyautogui.keyUp(keys)

    # This will make mario jump
    def jump(self):
        pyautogui.keyUp(keys)
        pyautogui.keyDown('up')

    # This will make mario walk left
    def left(self):
        pyautogui.keyUp(keys)
        pyautogui.keyDown('left')

    # This will make mario run left
    def runLeft(self):
        pyautogui.keyUp(keys)
        pyautogui.keyDown('left')
        pyautogui.keyDown('a')

    # This will make mario walk right
    def right(self):
        pyautogui.keyUp(keys)
        pyautogui.keyDown('right')

    # This will make mario run right
    def runRight(self):
        pyautogui.keyUp(keys)
        pyautogui.keyDown('right')
        pyautogui.keyDown('a')

    # This will make mario down (crouch if not in the air, or ground pound if in the air)
    def down(self):
        pyautogui.keyUp(keys)
        pyautogui.keyDown('down')

    # This will press on the screen where the button for the powerup normally is
    def powerUpTap(self):
        pass
