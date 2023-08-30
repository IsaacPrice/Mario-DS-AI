import keyboard
from Input import Input

class DebugInput:
    def __init__(self, Input, Mappings):
        self.Input = Input
        self.Mappings = Mappings
    
    def PollKeyboard(self):
        try:
            if keyboard.is_pressed(self.Mappings['left']) and not keyboard.is_pressed(self.Mappings['Sprint']):
                self.Input.walk_left()
            elif keyboard.is_pressed(self.Mappings['right']) and not keyboard.is_pressed(self.Mappings['Sprint']):
                self.Input.walk_right()
            elif keyboard.is_pressed(self.Mappings['left']) and keyboard.is_pressed(self.Mappings['Sprint']):
                self.Input.run_left()
            elif keyboard.is_pressed(self.Mappings['right']) and keyboard.is_pressed(self.Mappings['Sprint']):
                self.Input.run_right()
            if keyboard.is_pressed(self.Mappings['jump']):
                self.Input.jump()
        except:
            pass # Means that nothing was pressed