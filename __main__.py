import numpy as np
import time
import tensorflow as tf
from doubleDQN import DoubleDQN
from Input import Input
from DataProccesing import preprocess_image
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor

# Create the game
emu = DeSmuME()
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
saver.load_file('new_attempt/W1-1.sav')
inputs = Input(emu)
action_mapping = {
    0: inputs.none,
    1: inputs.walk_left,
    2: inputs.walk_right,
    3: inputs.run_left,
    4: inputs.run_right,
    5: inputs.jump,
    6: inputs.jump_left,
    7: inputs.jump_right
}

# AI & needed data
update_n_frames = 2
episodes = 1000
epsilon = 1
epsilon_min = 0.01
decay = .995
total_reward = 0
frames = 0
total_rewards = []
frame_stack = np.zeros((64, 48, 4))
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(64, 48, 4)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(alpha=0.001))
])
agent = DoubleDQN((64, 48, 4), 7, model_structure=model, epsilon=1)

# Debug data
frame_count = 0
start_time = time.time()

for e in range(episodes):
    reward = 0
    total_reward = 0
    frames = 0
    playing = True
    action = 0
    frame_stack = np.zeros((64, 48, 4))

    while playing:
        window.process_input()
        frames += 1

        # Update the current frame stack
        current_stack = frame_stack

        # Handle inputs
        if frames % update_n_frames == 0:
            try:
                action = agent.choose_action(current_stack)
            except: 
                pass
        action_mapping[action]()

        # Handle emulator cycles
        emu.cycle()
        window.draw()

        # Get the new screen state, and push out the last one
        frame = emu.screenshot()
        frame, dead = preprocess_image(frame)
        frame = frame.reshape(64, 48, 1)
        frame_stack = np.concatenate((frame_stack, frame), axis=2)
        frame_stack = frame_stack[:, :, 1:]

        # Calculate Reward
        reward += (emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000)
        total_reward += reward

        if frames % update_n_frames == 0:
            agent.learn(current_stack, action, reward, frame_stack)
            reward = 0
        
        # Calculate the frame rate
        frame_count += 1
        if frame_count % 60 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 60 / elapsed_time
            print(f"Frame Rate: {fps} FPS, Reward: {reward}, Total Reward: {total_reward}")
            start_time = time.time()
        
        # Cut off the AI if it goes for too long
        if frames > 10000:
            dead = True

        # Handle process when the AI dies
        if dead: 
            reward -= 3
            playing = False
            agent.learn(current_stack, action, reward, frame_stack)
            saver.load_file('W1-1.sav')
    
    # The episode is done
    if epsilon > epsilon_min:
        epsilon *= decay
        agent.set_epsilon(epsilon)
    
    # Adjust the online network
    agent.soft_update()
