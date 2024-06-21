import copy
import gym
import torch
import random

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple
from base64 import b64encode

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from gym.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit

from mario_env import MarioDSEnv
from memory_profiler import profile

import os
from pympler.tracker import SummaryTracker
from frameDisplay import FrameDisplay

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
num_gpus = torch.cuda.device_count()

import math
from torch.nn.init import kaiming_uniform_, zeros_

import torch
from collections import deque
import optuna

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
        obs, rews, dones, _, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, False, infos

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
    obs, rews, dones, _, infos = self.env.step(action)
    if not self.is_vector_env:
      rews = np.array([rews])
    self.returns = self.returns * self.gamma + rews
    rews = self.normalize(rews)
    self.returns[dones] = 0.0
    if not self.is_vector_env:
      rews = rews[0]
    
    return obs, rews, dones, False, infos
  
  def normalize(self, rews):
    self.return_rms.update(self.returns)
    return rews / np.sqrt(self.return_rms.var + self.epsilon)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        self.w_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.w_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        self.b_mu = nn.Parameter(torch.empty((out_features)))
        self.b_sigma = nn.Parameter(torch.empty((out_features)))

        kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        kaiming_uniform_(self.w_sigma, a=math.sqrt(5))
        zeros_(self.b_mu)
        zeros_(self.b_sigma)

    def forward(self, x, sigma=0.5):
        if self.training:  # Check if the model is in training mode
            w_noise = torch.normal(0, sigma, size=self.w_mu.size()).to(device)
            b_noise = torch.normal(0, sigma, size=self.b_mu.size()).to(device)
            return F.linear(x, self.w_mu + self.w_sigma * w_noise, self.b_mu + self.b_sigma * b_noise)
        else:
            return F.linear(x, self.w_mu, self.b_mu)
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class DQN(nn.Module):

  def __init__(self, hidden_size, num_layers, obs_shape, n_actions, sigma=0.5, atoms=51): 
    super(DQN, self).__init__()

    self.atoms = atoms
    self.n_actions = n_actions

    self.conv = nn.Sequential(
        ResidualBlock(obs_shape[0], 64, stride=2),
        ResidualBlock(64, 128, stride=2),
        ResidualBlock(128, 256, stride=2),
    )

    conv_out_size = 256 * 6 * 8
    self.head = nn.Sequential(
        NoisyLinear(conv_out_size, hidden_size, sigma=sigma),
        nn.ReLU(),
        NoisyLinear(hidden_size, hidden_size, sigma=sigma),
        nn.ReLU()
    )

    self.fc_adv = NoisyLinear(hidden_size, n_actions * self.atoms, sigma=sigma)
    self.fc_value = NoisyLinear(hidden_size, self.atoms, sigma=sigma)

  def _get_conv_out(self, shape):
    with torch.no_grad():
      conv_out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(conv_out.size()))
    
  def forward(self, x):
    x = x.to(device)
    x = self.conv(x)
    x= x.view(x.size(0), -1)
    x = self.head(x)
    adv = self.fc_adv(x).view(-1, self.n_actions, self.atoms)
    value = self.fc_value(x).view(-1, 1, self.atoms)
    q_logits = value + adv - adv.mean(dim=1, keepdim=True)
    q_probs = F.softmax(q_logits, dim=-1)
    return q_probs.to(device)

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

max_reward = -9999


class DeepQLearning(LightningModule):

  # Initialize
  def __init__(self, env_name, policy=greedy, capacity=10_000,
               batch_size=256, lr=1e-3, hidden_size=128, gamma=0.99,
               loss_fn=F.smooth_l1_loss, optim=AdamW, num_layers=2,
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

    self.q_net = DQN(hidden_size, num_layers, obs_shape, n_actions, sigma=sigma, atoms=atoms).to(device)
    self.target_q_net = copy.deepcopy(self.q_net).to(device)

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

      next_state, reward, done, _, info = self.env.step(action) 
      self.env.render()
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
    returns = returns.unsqueeze(1).to(device)
    dones = dones.unsqueeze(1).to(device)
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

    cross_entropies = -(m * log_action_value_probs).sum(dim=-1).to(device) # (batch_size, atoms) -> (batch_size)

    for idx, e in zip(indices, cross_entropies):
      self.buffer.update(idx, e.item())

    loss = (cross_entropies * weights.to(device)).mean()

    self.log('episode/Q-Error', loss)
    return loss

  # Training epoch end
  def training_epoch_end(self, training_step_outputs):
    global max_reward
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
    if self.env.return_queue[-1] > max_reward:
      max_reward = self.env.return_queue[-1]
    self.log('episode/Return', self.env.return_queue[-1])

    if self.current_epoch % self.hparams.sync_rate == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.best_score = None
      self.wait_count = 0
    
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.logged_metrics['episode/Return']
        print(logs)

trial_num = 1


def objective(trial):
  global trial_num, max_reward
  # Define the hyperparameters to tune
  lr = trial.suggest_float('lr', 1e-5, 1e-2)
  hidden_size = trial.suggest_int('hidden_size', 32, 512)
  num_layers = trial.suggest_int('num_layers', 1, 4)
  sync_rate = trial.suggest_int('sync_rate', 1, 10)
  a_start = trial.suggest_uniform('a_start', 0.6, 1.0)
  b_start = trial.suggest_uniform('b_start', 0.6, 1.0)
  a_end = trial.suggest_uniform('a_end', 0.1, 0.3)
  b_end = trial.suggest_uniform('b_end', 0.01, 0.2)
  a_last_episode = trial.suggest_int('a_last_episode', 500, 2000)
  b_last_episode = trial.suggest_int('b_last_episode', 500, 2000)
  sigma = trial.suggest_uniform('sigma', 0.1, 1.0)
  n_steps = trial.suggest_int('n_steps', 1, 20)

  print(f"Max Reward: {max_reward}\n\n\n")
  print(f"Trial {trial_num} Parameters:\n-------------------------")
  print("lr:", lr)
  print("hidden_size:", hidden_size)
  print("num_layers:", num_layers)
  print("sync_rate:", sync_rate)
  print("a_start:", a_start)
  print("b_start:", b_start)
  print("a_end:", a_end)
  print("b_end:", b_end)
  print("a_last_episode:", a_last_episode)
  print("b_last_episode:", b_last_episode)
  print(f"sigma: {sigma}")

  # Write trial details to a file
  with open('trial_details.txt', 'a') as file:
    file.write(f"Max Reward: {max_reward}\n\n\n")
    file.write(f"Trial {trial_num} Parameters:\n-------------------------\n")
    file.write(f"lr: {lr}\n")
    file.write(f"hidden_size: {hidden_size}\n")
    file.write(f"num_layers: {num_layers}\n")
    file.write(f"sync_rate: {sync_rate}\n")
    file.write(f"a_start: {a_start}\n")
    file.write(f"b_start: {b_start}\n")
    file.write(f"a_end: {a_end}\n")
    file.write(f"b_end: {b_end}\n")
    file.write(f"a_last_episode: {a_last_episode}\n")
    file.write(f"b_last_episode: {b_last_episode}\n")
    file.write(f"sigma: {sigma}\n")


  # Create the algorithm instance
  algo = DeepQLearning(
    'mario',
    lr=lr,
    hidden_size=hidden_size,
    num_layers=num_layers,
    sync_rate=sync_rate,
    a_start=a_start,
    b_start=b_start,
    a_end=a_end,
    b_end=b_end,
    a_last_episode=a_last_episode,
    b_last_episode=b_last_episode,
    sigma=sigma,
    frame_skip=10,
    save_movie_every=50,
    n_steps=n_steps
  )

  # Create the trainer instance
  trainer = Trainer(
    gpus=num_gpus,
    max_epochs=3_000,
    callbacks=[early_stop_callback]
  )

  max_reward = -999

  # Train the algorithm
  trainer.fit(algo)

  trial_num += 1

  # Return the negative return as the objective value (since Optuna minimizes the objective)
  print(trainer.logged_metrics['episode/Return'])
  return max_reward

"""
study = optuna.create_study(direction='maximize')
distributed_study = optuna_distributed.from_study(study)
distributed_study.optimize(objective, n_trials=100)

# Get the best hyperparameters
#best_params = distributed_study.best_params
best_return = distributed_study.best_value

print("Best Hyperparameters:")
print(best_params)
print("Best Return:", best_return)
"""


algo = DeepQLearning(
    'mario',
    lr=0.001,
    hidden_size=512,
    num_layers=2,
    sync_rate=5,
    a_last_episode=1000,
    b_last_episode=1000,
    sigma=.5,
    frame_skip=10,
    save_movie_every=5,
    n_steps=15
)

trainer = Trainer(
    max_epochs=10_000
)

trainer.fit(algo)