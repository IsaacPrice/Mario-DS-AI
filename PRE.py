import random
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.max_size = size
        self.alpha = 0.6  # determines how much prioritization is used

    def add(self, experience, error):
        priority = (error + -5) ** self.alpha
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace the experience with the lowest priority
            min_index = np.argmin(self.priorities)
            self.buffer[min_index] = experience
            self.priorities[min_index] = priority

    def sample(self, batch_size):
        probabilities = [p / sum(self.priorities) for p in self.priorities]
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices, errors):
        for index, error in zip(indices, errors):
            priority = (error + 1e-5) ** self.alpha
            self.priorities[index] = priority
    
    def __len__(self):
        return len(self.buffer)