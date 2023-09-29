import keyboard
from Input import Input

class DebugInput:
    def __init__(self, Input, Mappings):
        self.Input = Input
        self.Mappings = Mappings
    
    def poll_keyboard(self, Inputs):
        try:
            if keyboard.is_pressed(self.Mappings['left']) and not keyboard.is_pressed(self.Mappings['sprint']):
                return 4
            elif keyboard.is_pressed(self.Mappings['right']) and not keyboard.is_pressed(self.Mappings['sprint']):
                return 5
            elif keyboard.is_pressed(self.Mappings['left']) and keyboard.is_pressed(self.Mappings['sprint']):
                return 6
            elif keyboard.is_pressed(self.Mappings['right']) and keyboard.is_pressed(self.Mappings['sprint']):
                return 7
            if keyboard.is_pressed(self.Mappings['jump']):
                return 1
        except:
            pass
        
        return 0