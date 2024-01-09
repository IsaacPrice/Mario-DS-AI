import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.max_size = size
        self.alpha = alpha 

    def add(self, experience, error):
        # Ensure error is positive to avoid complex numbers
        priority = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace the experience with the lowest priority
            min_index = np.argmin(self.priorities)
            self.buffer[min_index] = experience
            self.priorities[min_index] = priority

    def sample(self, batch_size):
        # Calculate probabilities while ensuring they are real numbers
        total_priority = sum(self.priorities)
        probabilities = [p / total_priority for p in self.priorities]
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]

    def update_priorities(self, indices, errors):
        for index, error in zip(indices, errors):
            # Ensure error is positive to avoid complex numbers
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)