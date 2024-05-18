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

from pympler.tracker import SummaryTracker
from frameDisplay import FrameDisplay

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

import math
from torch.nn.init import kaiming_uniform, zeros_

import torch
from collections import deque

class Smoother:
    def __init__(self, n, device='cpu'):
        self.n = n
        self.device = device
        self.previous_tensors = deque(maxlen=n)
    
    def smooth(self, new_tensor):
        # Ensure the new tensor is on the correct device
        new_tensor = new_tensor.to(self.device)
        
        self.previous_tensors.append(new_tensor)

        # If this is the first tensor, just return it
        if len(self.previous_tensors) == 1:
            return new_tensor
        else:
            # Stack the tensors and calculate the mean for smoothing
            stacked_tensors = torch.stack(list(self.previous_tensors))
            smoothed_tensor = torch.mean(stacked_tensors, dim=0)
            return smoothed_tensor


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    def __init__(
        self,
        env,
        epsilon=1e-8,
    ):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos, _ = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        if not return_info:
            return obs
        else:
            return obs, info

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(gym.core.Wrapper):
  def __init__(
      self,
      env,
      gamma=0.99,
      epsilon=1e-8,
  ):
      super().__init__(env)
      self.num_envs = getattr(env, "num_envs", 1)
      self.is_vector_env = getattr(env, "is_vector_env", False)
      self.return_rms = RunningMeanStd(shape=())
      self.returns = np.zeros(self.num_envs)
      self.gamma = gamma
      self.epsilon = epsilon

  def step(self, action):
    # next_state, reward, done, info, _ = self.env.step(action) 
    obs, rews, infos, dones, _ = self.env.step(action)
    if not self.is_vector_env:
      rews = np.array([rews])
    self.returns = self.returns * self.gamma + rews
    rews = self.normalize(rews)
    self.returns[dones] = 0.0
    if not self.is_vector_env:
      rews = rews[0]
    
    return obs, rews, infos, dones, False
  
  def normalize(self, rews):
    self.return_rms.update(self.returns)
    return rews / np.sqrt(self.return_rms.var + self.epsilon)

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

  def __init__(self, hidden_size, obs_shape, n_actions, sigma=0.5, atoms=51): 
    super(DQN, self).__init__()

    self.atoms = atoms
    self.n_actions = n_actions

    self.conv = nn.Sequential(
        nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
    )

    conv_out_size = self._get_conv_out(obs_shape)
    self.head = nn.Sequential(
        NoisyLinear(conv_out_size, hidden_size, sigma=sigma),
        nn.ReLU(),
        NoisyLinear(hidden_size, hidden_size, sigma=sigma),
        nn.ReLU(),
    )

    self.fc_adv = NoisyLinear(hidden_size, n_actions * self.atoms, sigma=sigma)
    self.fc_value = NoisyLinear(hidden_size, self.atoms, sigma=sigma)

  def _get_conv_out(self, shape):
    with torch.no_grad():
      conv_out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(conv_out.size()))
    
  def forward(self, x):
    x = self.conv(x.float()).view(x.size()[0], -1) 
    x = self.head(x)
    adv = self.fc_adv(x).view(-1, self.n_actions, self.atoms)
    value = self.fc_value(x).view(-1, 1, self.atoms)
    q_logits = value + adv - adv.mean(dim=1, keepdim=True)
    q_probs = F.softmax(q_logits, dim=-1)
    return q_probs

def greedy(state, net, support):
  state = torch.tensor([state]).to(device)
  q_value_probs = net(state) # (1, n_actions, atoms)
  q_values = (q_value_probs * support).sum(-1) # (1, n_actions)
  action = torch.argmax(q_values, dim=-1) # (1, 1)
  action = int(action.item()) # ()
  return action, q_values

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


def create_environment(name, frame_skip, frame_stack=4):
  env = MarioDSEnv(frame_skip=frame_skip, frame_stack=frame_stack)
  env = NormalizeReward(env)
  env = RecordEpisodeStatistics(env)
  return env

smoother = Smoother(33)
frame_display = FrameDisplay(scale=4, num_q_values=8)


class DeepQLearning(LightningModule):

  # Initialize
  def __init__(self, env_name, policy=greedy, capacity=10_000,
               batch_size=256, lr=1e-3, hidden_size=128, gamma=0.99,
               loss_fn=F.smooth_l1_loss, optim=AdamW,
               samples_per_epoch=1_000, sync_rate=10,
               a_start = 0.5, a_end = 0.0, a_last_episode = 100,
               b_start = 0.4, b_end = 1.0, b_last_episode = 100,
               sigma=0.5, frame_skip=2, frame_stack=4, save_movie_every=50,
               n_steps=3, v_min=-10, v_max=10, atoms=51):

    super().__init__()

    self.support = torch.linspace(v_min, v_max, atoms, device=device)
    self.delta = (v_max - v_min) / (atoms - 1)

    self.env = create_environment(env_name, frame_skip=frame_skip)

    obs_shape = self.env.observation_space.shape
    n_actions = self.env.action_space.n

    self.q_net = DQN(hidden_size, obs_shape, n_actions, sigma=sigma, atoms=atoms)
    self.target_q_net = copy.deepcopy(self.q_net)

    self.policy = policy
    self.buffer = ReplayBuffer(capacity)

    self.save_hyperparameters()

    self.episode = 0
    self.movie_every = save_movie_every

    while len(self.buffer) < self.hparams.samples_per_epoch:
      print(f"{len(self.buffer)} samples in experience buffer. Filling.")
      self.play_episode()
    self.episode = 0

  @torch.no_grad()
  def play_episode(self, policy=None):
    if ((self.episode - 1) % self.movie_every == 0) and (policy is not None):
      state = self.env.reset(save_movie=True, episode=self.episode-1)
    else:
      state = self.env.reset()
    done = False
    transitions = []

    self.episode += 1

    q_values = torch.zeros(self.env.action_space.n)

    while not done:
      if policy:
        action, q_values = policy(state, self.q_net, self.support)
      else:
        action = self.env.action_space.sample()

      next_state, reward, infos, done, _ = self.env.step(action) 
      self.env.render()
      frame_display.display_frames(next_state, smoother.smooth(q_values.flatten()))
      exp = (state, action, reward, done, next_state)
      self.buffer.append(exp)
      state = next_state

      for i, (s, a, r, d, ns) in enumerate(transitions):
        batch = transitions[i: i + self.hparams.n_steps]
        ret = sum([t[2] * self.hparams.gamma**j for j, t in enumerate(batch)])
        _, _, _, ld, ls = batch[-1]
        self.buffer.append((s, a, ret, ld, ls))

  # Forward
  def forward(self, x):
    return self.q_net(x)

  # Configure Optimizers
  def configure_optimizers(self):
    q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
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
    indices, weights, states, actions, returns, dones, next_states = batch
    returns = returns.unsqueeze(1)
    dones = dones.unsqueeze(1)
    batch_size = len(indices)

    q_value_probs = self.q_net(states) # (batch_size, n_actions, atoms)
    action_value_probs = q_value_probs[range(batch_size), actions, :] # (batch_size, atoms)
    log_action_value_probs = torch.log(action_value_probs + 1e-6) # (batch_size, atoms)

    with torch.no_grad():
      next_q_value_probs = self.q_net(next_states) # (batch_size, n_actions, atoms)
      next_q_values = (next_q_value_probs * self.support).sum(dim=-1) # (batch_size, n_actions)
      next_actions = next_q_values.argmax(dim=-1) # (batch_size)

      next_q_value_probs = self.target_q_net(next_states) # (batch_size, n_actions, atoms)
      next_action_value_probs = next_q_value_probs[range(batch_size), next_actions, :] # (batch_size, atoms)

    m = torch.zeros(batch_size * self.hparams.atoms, device=device, dtype=torch.float64)

    Tz = returns + ~dones * self.hparams.gamma**self.hparams.n_steps * self.support.unsqueeze(0)
    Tz.clamp_(min=self.hparams.v_min, max=self.hparams.v_max)

    b = (Tz - self.hparams.v_min) / self.delta # (batch_size, atoms)
    l, u = b.floor().long(), b.ceil().long() # (batch_size, atoms)

    offset = torch.arange(batch_size, device=device).view(-1, 1) * self.hparams.atoms

    l_idx = (l + offset).flatten() # (batch_size * atoms)
    u_idx = (u + offset).flatten() # (batch_size * atoms)

    upper_probs = (next_action_value_probs * (u - b)).flatten().to(m.dtype) # (batch_size * atoms)
    lower_probs = (next_action_value_probs * (b - l)).flatten().to(m.dtype) # (batch_size * atoms)

    m.index_add_(dim=0, index=l_idx, source=upper_probs)
    m.index_add_(dim=0, index=u_idx, source=lower_probs)

    m = m.reshape(batch_size, self.hparams.atoms) # (batch_size, atoms)

    cross_entropies = -(m * log_action_value_probs).sum(dim=-1) # (batch_size, atoms) -> (batch_size)

    for idx, e in zip(indices, cross_entropies):
      self.buffer.update(idx, e.item())

    loss = (cross_entropies * weights).mean()

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


# Train the policy

algo = DeepQLearning(
  'mario', 
  lr=0.0005,
  hidden_size=512,
  sync_rate=3,
  a_end=0.2,
  b_end=0.4,
  a_last_episode=1500,
  b_last_episode=1500,
  sigma=0.8,
  frame_skip=10,
  save_movie_every=5,
  n_steps=15
)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs=100_000
)

trainer.fit(algo)

