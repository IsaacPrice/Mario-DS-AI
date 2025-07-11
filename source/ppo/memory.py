import torch
import numpy as np


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_probs = []
        self.old_values = []
    
    
    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.old_log_probs.append(log_prob)
        self.old_values.append(value)
    

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_log_probs = []
        self.old_values = []
    

    def get_tensors(self, device):
        frames = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)

        dones_array = np.array(self.dones, dtype=bool)
        dones = torch.tensor(dones_array, dtype=torch.bool).to(device)
        old_log_probs = torch.FloatTensor(self.old_log_probs).to(device)
        old_values = torch.FloatTensor(self.old_values).to(device)
        
        return frames, actions, rewards, dones, old_log_probs, old_values
    
