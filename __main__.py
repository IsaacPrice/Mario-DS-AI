import numpy as np
import time
from Input import Input
from DataProccesing import preprocess_image
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from newDQN import DQN
from PRE import PrioritizedReplayBuffer

# Create the game
emu = DeSmuME()
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
saver.load_file('W1-1.sav')
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
epsilon = 0.1
epsilon_min = 0.01
decay = .95
total_reward = 0
frames = 0
batch_size = 32
error = 100
total_rewards = []
frame_stack = np.zeros((64, 48, 4))
agent = DQN((64, 48, 4), 7, epsilon=epsilon)
replay_buffer = PrioritizedReplayBuffer(10000)


# Debug data
frame_count = 0

def learn_from_batch(experiences):
    """
    experiences: a list of (state, action, reward, next_state) tuples

    returns: a list of the new priorities for each experience
    """
    new_errors = []
    for experience in experiences:
        current_stack, action, reward, frame_stack = experience
        loss = agent.learn(current_stack, action, reward, frame_stack)
        
        # Assume loss represents the error or priority for updating the replay buffer
        new_errors.append(loss)

    return new_errors

def are_experiences_equal(exp1, exp2):
    """
    exp1 and exp2 are tuples of (state, action, reward, next_state)

    returns: True if the experiences are equal, False otherwise
    """
    if len(exp1) != len(exp2):
        return False
    for item1, item2 in zip(exp1, exp2):
        if isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            if not np.array_equal(item1, item2):
                return False
        elif item1 != item2:
            return False
    return True

def find_experience_index(buffer, experience):
    """
    buffer: a list of experiences
    experience: a tuple of (state, action, reward, next_state)

    returns: the index of the experience in the buffer, or -1 if it is not found
    """
    for index, buffer_experience in enumerate(buffer):
        if are_experiences_equal(buffer_experience, experience):
            return index
    return -1  # or raise an exception if the experience should always be in the buffer



for e in range(episodes):
    reward = 0
    total_reward = 0
    frames = 0
    playing = True
    action = 0
    frame_stack = np.zeros((64, 48, 4))
    start_time = time.time()
    agent.q_network.reset_noise()

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
            error = agent.learn(current_stack, action, reward, frame_stack)
            reward = 0
        
        # Calculate the frame rate
        frame_count += 1
        if frame_count % 60 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 60 / elapsed_time
            print(f"Frame Rate: {fps:.1f} FPS, Reward: {total_reward:.3f}, Epsilon: {epsilon:.2f}")
            start_time = time.time()
        
        # Cut off the AI if it goes for too long
        if frames > 14000:
            dead = True

        # Handle process when the AI dies
        if dead: 
            reward -= 3
            playing = False
            agent.learn(current_stack, action, reward, frame_stack)
            saver.load_file('W1-1.sav')
        
        agent.soft_update()
        replay_buffer.add((current_stack, action, reward, frame_stack), error)

    # ---------------------------------------------------------------------------- #
    
    # The episode is done
    if epsilon > epsilon_min:
        epsilon *= decay
        agent.set_epsilon(epsilon)
    
    # Adjust the online network
    agent.soft_update(.2)

    # Save the model
    if e % 25 == 0:
        agent.save(f"models/model-{e}.pt")

    
    # use the prioritized replay buffer
    if len(replay_buffer) >= batch_size:
        experiences = replay_buffer.sample(batch_size)
        new_errors = learn_from_batch(experiences)


        # update the priorities
        try:
            indices = [find_experience_index(replay_buffer.buffer, exp) for exp in experiences]
        except ValueError as e:
            print("Error finding experience in buffer:", e)
            # Optionally, print the problematic experience and buffer state

        replay_buffer.update_priorities(indices, new_errors)

