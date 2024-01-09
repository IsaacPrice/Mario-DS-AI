import numpy as np
import torch
import time
from Input import Input
from DataProccesing import preprocess_image_tensor
from desmume.emulator import DeSmuME, DeSmuME_Savestate, DeSmuME_Memory, MemoryAccessor
from newDQN import DQN
from PRE import PrioritizedReplayBuffer
from windows_sleep import *
from frameDisplay import FrameDisplay

# Create the game
emu = DeSmuME()
emu.open('NSMB.nds')
window = emu.create_sdl_window()
saver = DeSmuME_Savestate(emu)
saver.load_file('W1-3.sav')
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
episodes = 1000
epsilon = .1
epsilon_min = 0.01
decay = .995
total_reward = 0
frames = 0
batch_size = 64
error = 100
total_rewards = []
agent = DQN((4, 96, 128), 8, epsilon=epsilon)
replay_buffer = PrioritizedReplayBuffer(1000)

try:
    print("Preventing sleep")
    prevent_sleep()
except:
    print("Failed to prevent sleep")
    allow_sleep()

#agent.load("models/model-200.pt")

# Debug data
frame_count = 0

class RewardTracker:
    def __init__(self, n=600):
        # Initialize the array with n zeros
        self.rewards = np.zeros(n)

    def add_reward(self, reward):
        # Add a new reward to the start of the array and remove the last one
        self.rewards = np.insert(self.rewards, 0, reward)[:-1]

    def sum_rewards(self):
        # Return the sum of all the rewards
        return np.sum(self.rewards)

    def reset_rewards(self):
        # Reset the rewards to all zeros
        self.rewards = np.zeros(len(self.rewards))

class QValueHistory:
    def __init__(self, capacity=7):
        self.capacity = capacity
        self.history = []

    def add(self, q_values):
        if len(self.history) == self.capacity:
            self.history.pop(0)
        self.history.append(q_values)

    def average(self):
        if not self.history:
            return torch.tensor([])
        
        # Ensuring all tensors are on the same device before averaging
        device = self.history[0].device
        tensors_on_same_device = [q.to(device) for q in self.history]
        return torch.stack(tensors_on_same_device).mean(dim=0)

    def __lt__(self, other):
        return self.average() < other

    def __le__(self, other):
        return self.average() <= other

    def __gt__(self, other):
        return self.average() > other

    def __ge__(self, other):
        return self.average() >= other

    def __eq__(self, other):
        return torch.equal(self.average(), other)

    def __ne__(self, other):
        return not torch.equal(self.average(), other)

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
        elif torch.is_tensor(item1) and torch.is_tensor(item2):
            if not torch.equal(item1, item2):
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

frameDisplay = FrameDisplay(frame_shape=(96, 128), scale=2, spacing=5, window_size=(600, 1200), num_q_values=8)
q_valueHistory = QValueHistory(5)
rewardTracker = RewardTracker()

for e in range(episodes):
    reward = 0
    total_reward = 0
    frames = 0
    playing = True
    action = 0
    start_time = time.time()
    agent.q_network.reset_noise()
    print(f"Episode {e}, Epsilon {epsilon:.2f}", end="")
    tested = torch.zeros(8)

    # Initialize frame stack as a PyTorch tensor
    frame_stack = torch.zeros((4, 96, 128), dtype=torch.float16)  # Assuming height=48, width=64
    if torch.cuda.is_available():
        frame_stack = frame_stack.cuda()

    while playing:
        frames += 1

        # Update the current frame stack
        current_stack = frame_stack

        # Handle inputs
        try:
            distribution = agent.choose_action(current_stack)
            action = distribution.max(1)[1].item()
        except: 
            pass
        action_mapping[action]()

        # Handle emulator cycles
        emu.cycle()
        window.draw()

        # Get the new screen state, and push out the last one
        frame = emu.screenshot()
        frame_tensor, dead = preprocess_image_tensor(frame)
        if torch.cuda.is_available():
            frame_tensor = frame_tensor.to('cuda')
            frame_stack = frame_stack.to('cuda')

        # Update the frame stack
        frame_stack = torch.cat((frame_stack, frame_tensor), dim=0)
        frame_stack = frame_stack[1:, :, :]  # Remove the oldest frame

        try:
            something = distribution.reshape(8)
            tested = something.detach()
        except:
            pass
        q_valueHistory.add(tested)
        frameDisplay.display_frames(frame_stack.cpu(), q_valueHistory.average().cpu())

        # Calculate Reward
        reward = (emu.memory.signed[0x021B6A90:0x021B6A90:4] / 20000) - 0.02
        rewardTracker.add_reward(reward)
        total_reward += reward
        
        # Calculate the frame rate
        frame_count += 1
        if frame_count % 60 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 60 / elapsed_time
            start_time = time.time()
        
        
        # Cut off the AI if it goes for too long
        if frames > 10000:
            dead = True

        # Handle process when the AI dies
        if dead: 
            reward -= 3
            playing = False
            #agent.learn(current_stack, action, reward, frame_stack)
        
        print(f"Reward: {rewardTracker.sum_rewards():.2f}", end="\r", flush=True)
        if rewardTracker.sum_rewards() < -3:
            agent.reset_noise()
            rewardTracker.reset_rewards()

        
        #agent.soft_update()
        replay_buffer.add((current_stack, action, reward, frame_stack), error)
        reward = 0

    # ---------------------------------------------------------------------------- #
        
    print(f", Reward {total_reward:.3f}")
    
    # The episode is done
    if epsilon > epsilon_min:
        epsilon *= decay
        agent.set_epsilon(epsilon)

    # Save the model
    if e % 25 == 0:
        agent.save(f"models/model-{e}.pt")

    
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

    # Adjust the online network
    agent.soft_update(.5)
    agent.reset_noise()
    saver.load_file('W1-3.sav')
    rewardTracker.reset_rewards()
    
frameDisplay.close()
allow_sleep()