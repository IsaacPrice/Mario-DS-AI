from desmume.controls import Keys, keymask

class Input:
    def __init__(self, emu, action_repeat_frames=3):
        self.emu = emu
        self.keys = [Keys.KEY_A, Keys.KEY_X, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_DOWN, Keys.KEY_UP]
        self.action_duration = 0  # Tracks how many frames to hold current action
        self.current_keys = []    # Tracks which keys are currently being held
        self.frame_persist_counter = 0  # Counter for frame persistence
        self.frame_persist_limit = action_repeat_frames    # Hold input for N frames
        self.current_binary_input = [0, 0, 0, 0, 0, 0]  # [UP, DOWN, LEFT, RIGHT, X, A]


    def release_all(self):
        for key in self.keys:
            self.emu.input.keypad_rm_key(keymask(key))
        self.current_keys = []
        self.action_duration = 0
        self.current_binary_input = [0, 0, 0, 0, 0, 0]


    def set_binary_input(self, binary_input):
        if self.frame_persist_counter >= self.frame_persist_limit or self.frame_persist_counter == 0:
            self.current_binary_input = binary_input.copy()
            self.frame_persist_counter = 0
            
            for key in self.keys:
                self.emu.input.keypad_rm_key(keymask(key))
            
            key_mapping = [Keys.KEY_UP, Keys.KEY_DOWN, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_X, Keys.KEY_A]
            self.current_keys = []
            
            for i, pressed in enumerate(binary_input):
                if pressed:
                    self.emu.input.keypad_add_key(keymask(key_mapping[i]))
                    self.current_keys.append(key_mapping[i])
        
        self.frame_persist_counter += 1

    def get_current_binary_input(self):
        return self.current_binary_input.copy()


    def execute_action(self):
        pass


    def set_action(self, keys_to_press, duration=1):
        """Set an action to be held for a specific duration"""
        self.release_all()
        self.current_keys = keys_to_press
        self.action_duration = duration
        
        for key in keys_to_press:
            self.emu.input.keypad_add_key(keymask(key))

    # Legacy action methods for backward compatibility
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