import copy
import gym
import torch
import random

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.callbacks import EarlyStopping

from gym.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit

from mario_env import MarioDSEnv
from memory_profiler import profile

import os
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

import math
from torch.nn.init import kaiming_uniform, zeros_

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, sigma):
    super(NoisyLinear, self).__init__()

    self.w_mu = nn.Parameter(torch.empty((out_features, in_features)))
    self.w_sigma = nn.Parameter(torch.empty((out_features, in_features)))
    self.b_mu = nn.Parameter(torch.empty((out_features)))
    self.b_sigma = nn.Parameter(torch.empty((out_features)))

    kaiming_uniform(self.w_mu, a=math.sqrt(5))
    kaiming_uniform(self.w_sigma, a=math.sqrt(5))
    zeros_(self.b_mu)
    zeros_(self.b_sigma)

  def forward(self, x, sigma=0.5):
    if self.training:
      w_noise = torch.normal(0, sigma, size=self.w_mu.size()).to(device)
      b_noise = torch.normal(0, sigma, size=self.b_mu.size()).to(device)
      return F.linear(x, self.w_mu + self.w_sigma * w_noise, self.b_mu + self.b_sigma * b_noise)
    else:
      return F.linear(x, self.w_mu, self.b_mu)

class DQN(nn.Module):

  def __init__(self, hidden_size, obs_shape, n_actions, sigma): 
    super(DQN, self).__init__()
    # Process visual information
    self.conv = nn.Sequential(
        nn.Conv2d(obs_shape[0], 64, kernel_size=3),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.MaxPool2d(kernel_size=4),
        nn.ReLU(),
    )

    conv_out_size = self._get_conv_out(obs_shape)
    self.head = nn.Sequential(
        NoisyLinear(conv_out_size, hidden_size, sigma=sigma),
        nn.ReLU(),
        NoisyLinear(hidden_size, hidden_size, sigma=sigma),
        nn.ReLU(),
    )

    self.fc_adv = NoisyLinear(hidden_size, n_actions, sigma=sigma)
    self.fc_value = NoisyLinear(hidden_size, 1, sigma=sigma)

  def _get_conv_out(self, shape):
    conv_out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(conv_out.size()))

    
  def forward(self, x):
    x = self.conv(x.float()).view(x.size()[0], -1) 
    x = self.head(x)
    adv = self.fc_adv(x)
    value = self.fc_value(x)
    return value + adv - torch.mean(adv, dim=1, keepdim=True)

def greedy(state, net):
  state = torch.tensor([state]).to(device)
  q_values = net(state)
  action = q_values.argmax(dim=-1)
  action = int(action.item())
  return action

class ReplayBuffer:

  # Constructor
  def __init__(self, capacity):
    self.buffer = deque(maxlen=capacity)
    self.priorities = deque(maxlen=capacity)
    self.capacity = capacity
    self.alpha = 1.0
    self.beta = 0.5
    self.max_priority = 0.0

  # __len__
  def __len__(self):
    return len(self.buffer)

  # Append
  def append(self, experience):
    self.buffer.append(experience)
    self.priorities.append(self.max_priority)

  # Update
  def update(self, index, priority):
    if priority > self.max_priority:
      self.max_priority = priority
    self.priorities[index] = priority

  # Sample
  def sample(self, batch_size):
    prios = np.array(self.priorities, dtype=np.float64) + 1e-4
    prios = prios ** self.alpha
    probs = prios / prios.sum()

    weights = (self.__len__() * probs) ** -self.beta
    weights = weights / weights.max()

    idx = np.random.choice(range(self.__len__()), p=probs, size=batch_size)
    sample = [(i, weights[i], *self.buffer[i]) for i in idx]

    return sample

class RLDataset(IterableDataset):
   def __init__(self, buffer, sample_size=200):
    self.buffer = buffer
    self.sample_size = sample_size

   def __iter__(self):
    for experience in self.buffer.sample(self.sample_size):
      yield experience


def create_environment(name):
  env = MarioDSEnv()
  env = RecordEpisodeStatistics(env)
  #env = NormalizeObservation(env)
  #env = NormalizeReward(env)
  return env


"""#### Create the Deep Q-Learning algorithm"""

class DeepQLearning(LightningModule):

  # Initialize
  def __init__(self, env_name, policy=greedy, capacity=100_000,
               batch_size=256, lr=1e-3, hidden_size=128, gamma=0.99,
               loss_fn=F.smooth_l1_loss, optim=AdamW,
               samples_per_epoch=10_000, sync_rate=10,
               a_start = 0.5, a_end = 0.0, a_last_episode = 100,
               b_start = 0.4, b_end = 1.0, b_last_episode = 100,
               sigma=0.5):

    super().__init__()
    self.env = create_environment(env_name)

    obs_shape = self.env.observation_space.shape
    n_actions = self.env.action_space.n
    self.q_net = DQN(hidden_size, obs_shape, n_actions, sigma=sigma)
    self.target_q_net = copy.deepcopy(self.q_net)

    self.policy = policy
    self.buffer = ReplayBuffer(capacity)

    self.save_hyperparameters()

    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f"{len(self.buffer)} samples in experience buffer. Filling.")
      self.play_episode()

  @torch.no_grad()
  def play_episode(self, policy=None):
    state = self.env.reset()
    done = False

    while not done:
      if policy:
        action = policy(state, self.env, self.q_net)
      else:
        action = self.env.action_space.sample()

      next_state, reward, done, info, _ = self.env.step(action) 
      self.env.render()
      exp = (state, action, reward, done, next_state)
      self.buffer.append(exp)
      state = next_state

  # Forward
  def forward(self, x):
    return self.q_net(x)


  # Configure Optimizers
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(),
                                         lr=self.hparams.lr)
    return [q_net_optimizer]


  # Create dataloader
  def train_dataloader(self):
    dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=self.hparams.batch_size
    )
    return dataloader


  # Training step
  def training_step(self, batch, batch_idx):
    indices, weights, states, actions, rewards, dones, next_states = batch
    weights = weights.unsqueeze(1)
    actions = actions.unsqueeze(1)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)

    state_action_values = self.q_net(states).gather(1, actions)

    with torch.no_grad():
      _, next_actions = self.q_net(next_states).max(dim=1, keepdim=True)
      next_action_values = self.target_q_net(next_states).gather(1, next_actions)
      next_action_values[dones] = 0.0

    expected_state_action_values = rewards + self.hparams.gamma * next_action_values

    # Compute the priorities and update
    td_errors = (state_action_values - expected_state_action_values).abs().detach()
    for idx, e in zip(indices, td_errors):
      self.buffer.update(idx, e.item())

    loss = weights * self.hparams.loss_fn(state_action_values, expected_state_action_values, reduction='none')
    loss = loss.mean()

    self.log('episode/Q-Error', loss)
    return loss

  # Training epoch end
  def training_epoch_end(self, training_step_outputs):
    alpha = max(
        self.hparams.a_end,
        self.hparams.a_start - self.current_epoch /
        self.hparams.a_last_episode
    )
    beta = min(
        self.hparams.b_end,
        self.hparams.b_start - self.current_epoch /
        self.hparams.b_last_episode
    )
    self.buffer.alpha = alpha
    self.buffer.beta = beta

    self.play_episode(policy=self.policy)
    self.log('episode/Return', self.env.return_queue[-1])

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())


"""#### Train the policy"""

algo = DeepQLearning(
  'mario', 
  hidden_size=512,
  sync_rate=3
)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs=4_500
)


trainer.fit(algo)

