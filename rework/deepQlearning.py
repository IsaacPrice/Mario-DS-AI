import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning

from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from mario_environment import MarioDSEnv
from custom_wrappers import NormalizeReward
from replay_buffer import ReplayBuffer, RLDataset
from DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy(state, net, support):
  q_value_probs = net(state.unsqueeze(0)) 
  q_values = (q_value_probs * support).sum(-1) 
  action = torch.argmax(q_values, dim=-1) 
  action = int(action.item()) 
  return action

def CreateEnvironment(envName: str = 'Mario'):
    if envName == 'Mario':
        env = MarioDSEnv(10, 4)
        env = NormalizeReward(env)
        return env

class DeepQLearning(LightningModule):

    def __init__(self, envName='Mario', policy=greedy, bufferCapacity=10_000, samples_per_epoch=1_000, batch_size=512, lr=1e-3,  hiddenSize=512, numLayers=2, gamma=0.99, lossFn=F.smooth_l1_loss, syncRate=10, sigma=0.5, a_start=0.5, a_end=0.0, a_last_episode=100, b_start=0.4, b_end=1.0, b_last_episode=100, atoms=51, v_min=-10, v_max=10, n_steps=15):
        super(DeepQLearning, self).__init__()

        self.env = CreateEnvironment(envName)
        self.policy = policy
        self.buffer = ReplayBuffer(bufferCapacity)
        self.q_net = DQN(hiddenSize, numLayers, self.env.observation_space.shape, self.env.action_space.n, sigma, atoms).to(device)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.support = torch.linspace(v_min, v_max, atoms, device=device)
        self.delta = (v_max - v_min) / (atoms - 1)

        self.save_hyperparameters()

        while len(self.buffer) < samples_per_epoch:
            print(f'Filling buffer: {len(self.buffer)}')
            self.play_episode()
        
        self.episode = 0

    @torch.no_grad()
    def play_episode(self, policy=None):
        state = self.env.reset()
        done = False
        transitions = []

        self.episode = 0

        while not done:
            if policy:
                action = policy(state, self.q_net, self.support)
            else: 
                action = self.env.action_space.sample()
            
            next_state, reward, done, *_ = self.env.step(action)
            exp = (state, action, reward, done, next_state)
            transitions.append(exp)
            state = next_state

            self.env.render()

        for i, (s, a, r, d, ns) in enumerate(transitions):
            batch = transitions[i: i + self.hparams.n_steps]
            ret = sum([t[2] * self.hparams.gamma**j for j, t in enumerate(batch)])
            _, _, _, ld, ls = batch[-1]
            self.buffer.append((s, a, ret, ld, ls))
    
    def forward(self, x):
        return self.q_net(x)
    
    def configure_optimizers(self):
        optimizer = Adam(self.q_net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size
        )
        return dataloader

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

        if self.episode % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
