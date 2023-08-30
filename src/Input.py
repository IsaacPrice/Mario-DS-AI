from desmume.controls import Keys, keymask

class Input:
    def __init__(self, emu):
        self.emu = emu
        self.keys = [Keys.KEY_A, Keys.KEY_X, Keys.KEY_LEFT, Keys.KEY_RIGHT, Keys.KEY_DOWN, Keys.KEY_UP]

    def release_all(self):
        for key in self.keys:
            self.emu.input.keypad_rm_key(keymask(key))

    def none(self):
        self.release_all()

    def jump(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_A))

    def jump_left(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_A))
        self.emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))

    def jump_right(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_A))
        self.emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

    def walk_left(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))

    def walk_right(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

    def run_left(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_X))
        self.emu.input.keypad_add_key(keymask(Keys.KEY_LEFT))

    def run_right(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_X))
        self.emu.input.keypad_add_key(keymask(Keys.KEY_RIGHT))

    def down(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_DOWN))

    def up(self):
        self.release_all()
        self.emu.input.keypad_add_key(keymask(Keys.KEY_UP))