# Mario DS AI - Reinforcement Learning

Train AI agents to play New Super Mario Bros. DS using advanced reinforcement learning algorithms. This project implements both Rainbow DQN and PPO with real-time visualization and automatic model saving.

## Features

- **Rainbow DQN**: Advanced Deep Q-Network with prioritized replay, dueling architecture, distributional RL, multi-step learning, and noisy networks
- **PPO**: Proximal Policy Optimization with actor-critic architecture and generalized advantage estimation
- **Real-time training visualization**: Live plots of rewards, losses, and performance metrics
- **Automatic model checkpointing**: Best models saved automatically with periodic backups
- **Episode recording**: MP4 videos generated for performance analysis

## Prerequisites

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Place the `NSMB.nds` ROM file in the project root directory
3. The DeSmuME emulator Python bindings will be installed automatically via requirements

## Quick Start

```bash
# Train PPO (recommended)
python train_mario.py --algorithm ppo --mode train --episodes 500

# Train Rainbow DQN
python train_mario.py --algorithm rainbow --mode train --episodes 500

# Test a trained model
python train_mario.py --algorithm ppo --mode test --model_path models/ppo_best.pth
```

## Usage

### Training Parameters
- `--algorithm ppo`: Uses Proximal Policy Optimization (generally more stable and faster)
- `--algorithm rainbow`: Uses Rainbow DQN (more exploratory, potentially higher performance ceiling)
- `--mode train`: Train a new model from scratch with live visualization
- `--mode test`: Load and evaluate a pre-trained model
- `--episodes N`: Number of episodes to run (default: 1000 for training, 5 for testing)
- `--save_interval N`: Save model checkpoint every N episodes (default: 100)
- `--frame_skip N`: Skip N frames between actions for faster training (default: 4)
- `--frame_stack N`: Stack N consecutive frames for temporal information (default: 4)

### Example Commands

```bash
# Long training session with frequent saves
python train_mario.py --algorithm ppo --episodes 2000 --save_interval 50

# Quick test with custom frame settings
python train_mario.py --algorithm ppo --mode test --model_path models/ppo_episode_500.pth --frame_skip 8 --frame_stack 3

# Fast training with higher frame skip
python train_mario.py --algorithm rainbow --episodes 1000 --frame_skip 10
```

## Training Process

During training, you'll see:
- **Real-time plots**: Episode rewards, training losses, and algorithm-specific metrics
- **Progress updates**: Current episode, best score, recent performance averages
- **Automatic saving**: Best models saved when new high scores are achieved
- **Video generation**: Episode recordings saved to `episodes/` directory

Models are saved in `models/` directory:
- `ppo_best.pth` / `rainbow_best.pth`: Best performing models
- `ppo_episode_N.pth` / `rainbow_episode_N.pth`: Periodic checkpoints

## Hyperparameter Tuning

The algorithms use optimized hyperparameters by default, but you can modify them in the source code:

**PPO**: Learning rate 0.0003, GAE lambda 0.95, clip epsilon 0.2, 4 training epochs per update
**Rainbow DQN**: Learning rate 0.0001, epsilon decay 0.995, 3-step returns, 51 distributional atoms

## Contributing

Feel free to experiment with:
- Hyperparameter tuning for different performance characteristics
- Custom reward functions for specific behaviors
- Additional RL algorithms (A3C, SAC, etc.)
- Environment modifications and level selection
- Enhanced visualization and analysis tools
