from Controls import *

inputs = Input()

while True:
    move = input()

    if move == 'left':
        inputs.left()
    elif move == 'right':
        inputs.right()
    elif move == 'runLeft':
        inputs.runLeft()
    elif move == 'runRight':
        inputs.runRight()
    elif move == 'jump':
        inputs.jump()
    elif move == 'none':
        inputs.none()
    