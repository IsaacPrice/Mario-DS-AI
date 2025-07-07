# Mario DS AI - Reinforcement Learning

This project implements two state-of-the-art reinforcement learning algorithms to play Mario DS using a Nintendo DS emulator.

## Features

- **Rainbow DQN**: Deep Q-Network with prioritized experience replay, dueling architecture, distributional RL, multi-step learning, and noisy networks
- **PPO**: Proximal Policy Optimization with generalized advantage estimation
- **Real-time visualization**: Live plots showing training progress, rewards, and losses
- **Episode recording**: Automatic video generation of episodes
- **Model saving/loading**: Checkpoint system for training resumption and testing

## Algorithms Implemented

### Rainbow DQN
- Prioritized Experience Replay
- Dueling Network Architecture
- Distributional RL (C51)
- Multi-step Learning
- Noisy Networks for exploration
- Double DQN

### PPO (Proximal Policy Optimization)
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Clipped policy gradients
- Entropy regularization

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the DeSmuME emulator Python bindings installed
3. Ensure you have the NSMB.nds ROM file and save states in the project directory

## Usage

### Interactive Launcher
```bash
python launcher.py
```
This will guide you through algorithm selection and training options.

### Command Line Training

#### Train Rainbow DQN:
```bash
python train_mario.py --algorithm rainbow --mode train --episodes 1000
```

#### Train PPO:
```bash
python train_mario.py --algorithm ppo --mode train --episodes 1000
```

#### Test a trained model:
```bash
python train_mario.py --algorithm rainbow --mode test --model_path models/rainbow_best.pth --episodes 5
```

### Quick Start Options
```bash
# Quick Rainbow DQN training
python launcher.py rainbow

# Quick PPO training
python launcher.py ppo

# Install requirements only
python launcher.py install
```

## Command Line Arguments

- `--algorithm`: Choose 'rainbow' or 'ppo'
- `--mode`: Choose 'train' or 'test'
- `--episodes`: Number of episodes (default: 1000 for training, 5 for testing)
- `--save_interval`: Save model every N episodes (default: 100)
- `--model_path`: Path to model file for testing
- `--frame_skip`: Number of frames to skip (default: 4)
- `--frame_stack`: Number of frames to stack (default: 4)

## File Structure

```
Mario-DS-AI/
├── mario_env.py          # Custom Gym environment for Mario DS
├── rainbow_dqn.py        # Rainbow DQN implementation
├── ppo_agent.py          # PPO implementation
├── train_mario.py        # Main training script
├── launcher.py           # Interactive launcher
├── requirements.txt      # Python dependencies
├── DataProccesing.py     # Image preprocessing utilities
├── Input.py              # Emulator input handling
├── NSMB.nds             # Nintendo DS ROM file
├── W1-1 (linux).dsv    # Save state file
├── models/              # Saved models directory
└── episodes/            # Episode videos directory
```

## Environment Details

The Mario DS environment provides:
- **Observation Space**: (4, 48, 64) - 4 stacked grayscale frames of 48x64 pixels
- **Action Space**: 8 discrete actions (none, walk left/right, run left/right, jump, jump left/right)
- **Reward Function**: Based on Mario's position progression with death penalty
- **Episode Termination**: Death detection or maximum steps (3000)

## Training Features

### Real-time Visualization
Both algorithms display live training metrics:
- Episode rewards over time
- Training losses
- Recent performance averages
- Algorithm-specific metrics (epsilon for DQN, policy/value losses for PPO)

### Model Saving
- Best models are automatically saved when achieving new high scores
- Periodic checkpoints saved every N episodes
- Episode videos generated for analysis

### Performance Monitoring
- Episode reward tracking
- Training loss monitoring
- Memory usage tracking (via pympler)
- GPU utilization (when available)

## Hyperparameters

### Rainbow DQN
- Learning rate: 0.0001
- Gamma: 0.99
- Epsilon decay: 0.995
- Batch size: 32
- Target network update: 1000 steps
- Multi-step: 3
- Distributional atoms: 51

### PPO
- Learning rate: 0.0003
- Gamma: 0.99
- Epsilon clip: 0.2
- K epochs: 4
- GAE lambda: 0.95
- Update timestep: 2048

## Troubleshooting

1. **CUDA Issues**: The code automatically detects GPU availability. If you have CUDA issues, it will fall back to CPU.

2. **Memory Issues**: If you encounter memory issues, try reducing:
   - Batch size
   - Buffer size (for Rainbow DQN)
   - Update timestep (for PPO)

3. **Emulator Issues**: Ensure DeSmuME Python bindings are properly installed and the ROM file is accessible.

4. **Dependencies**: Run `python launcher.py install` to install all required packages.

## Results

Models will be saved in the `models/` directory:
- `rainbow_best.pth` / `ppo_best.pth`: Best performing models
- `rainbow_episode_N.pth` / `ppo_episode_N.pth`: Periodic checkpoints

Episode videos will be saved in the `episodes/` directory as MP4 files.

## Contributing

Feel free to experiment with:
- Hyperparameter tuning
- Reward function modifications
- Additional RL algorithms
- Environment enhancements
- Visualization improvements
