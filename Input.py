from desmume.controls import Keys, keymask

class Input:
    def __init__(self, emu):
        self.emu = emu
        self.keys = [Keys.KEY_A, Keys.KEY_X, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_DOWN, Keys.KEY_UP]
        self.action_duration = 0  # Tracks how many frames to hold current action
        self.current_keys = []    # Tracks which keys are currently being held

    def release_all(self):
        for key in self.keys:
            self.emu.input.keypad_rm_key(keymask(key))
        self.current_keys = []
        self.action_duration = 0

    def execute_action(self):
        """Call this every frame to handle multi-frame actions"""
        if self.action_duration > 0:
            # Continue holding current keys
            self.action_duration -= 1
            if self.action_duration == 0:
                self.release_all()
        
    def set_action(self, keys_to_press, duration=1):
        """Set an action to be held for a specific duration"""
        self.release_all()
        self.current_keys = keys_to_press
        self.action_duration = duration
        
        for key in keys_to_press:
            self.emu.input.keypad_add_key(keymask(key))

    def none(self):
        self.set_action([], 1)

    def jump(self):
        self.set_action([Keys.KEY_A], 1)

    def jump_left(self):
        self.set_action([Keys.KEY_A, Keys.KEY_LEFT], 1)

    def jump_right(self):
        self.set_action([Keys.KEY_A, Keys.KEY_RIGHT], 1)

    def walk_left(self):
        self.set_action([Keys.KEY_LEFT], 1)

    def walk_right(self):
        self.set_action([Keys.KEY_RIGHT], 1)

    def run_left(self):
        self.set_action([Keys.KEY_X, Keys.KEY_LEFT], 1)

    def run_right(self):
        self.set_action([Keys.KEY_X, Keys.KEY_RIGHT], 1)

    def down(self):
        self.set_action([Keys.KEY_DOWN], 1)

    def up(self):
        self.set_action([Keys.KEY_UP], 1)
    
    # Extended jump actions for tall jumps - THESE HOLD FOR MULTIPLE FRAMES
    def hold_jump_right_short(self):
        """Hold jump+right for 3 frames (short sustained jump)"""
        self.set_action([Keys.KEY_A, Keys.KEY_RIGHT], 3)
    
    def hold_jump_right_medium(self):
        """Hold jump+right for 4 frames (medium sustained jump)"""
        self.set_action([Keys.KEY_A, Keys.KEY_RIGHT], 4)
    
    def hold_jump_right_long(self):
        """Hold jump+right for 5 frames (long/tall sustained jump)"""
        self.set_action([Keys.KEY_A, Keys.KEY_RIGHT], 5)
    
    def hold_jump_long(self):
        """Hold jump for 5 frames (tall jump on spot)"""
        self.set_action([Keys.KEY_A], 5)
    
    def run_jump_right(self):
        """Running jump right (run + jump together) for 3 frames"""
        self.set_action([Keys.KEY_X, Keys.KEY_A, Keys.KEY_RIGHT], 3)
    
    def run_jump_right_long(self):
        """Running jump right (run + jump together) for 5 frames - for tall obstacles"""
        self.set_action([Keys.KEY_X, Keys.KEY_A, Keys.KEY_RIGHT], 5)
    
    # Additional backward movement actions for strategic positioning
    def hold_jump_left_medium(self):
        """Hold jump+left for 4 frames (medium sustained jump backward)"""
        self.set_action([Keys.KEY_A, Keys.KEY_LEFT], 4)
    
    def run_jump_left(self):
        """Running jump left (run + jump together) for 3 frames - for backing up with momentum"""
        self.set_action([Keys.KEY_X, Keys.KEY_A, Keys.KEY_LEFT], 3)