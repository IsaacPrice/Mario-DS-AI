import argparse
import os
import glob
import re
from source.env.mario_env import MarioDSEnv
from source.dqn.rainbow_dqn import RainbowDQNAgent
from source.ppo.agent import PPOAgent


def find_latest_model(algorithm):
    episode_models = glob.glob(f"models/{algorithm}_episode_*.pth")
    if not episode_models:
        return None
    
    episode_numbers = []
    for model in episode_models:
        match = re.search(rf'{algorithm}_episode_(\d+)\.pth', model)
        if match:
            episode_numbers.append((int(match.group(1)), model))
    
    if not episode_numbers:
        return None
    
    return max(episode_numbers, key=lambda x: x[0])[1]


def load_model_if_specified(agent, algorithm, load_model=None, load_best=False, load_latest=False):
    model_path = None
    
    if load_model:
        model_path = load_model
        if not os.path.exists(model_path):
            print(f"❌ Specified model not found: {model_path}")
            return False
    elif load_best:
        model_path = f"models/{algorithm}_best.pth"
        if not os.path.exists(model_path):
            print(f"❌ Best model not found: {model_path}")
            return False
    elif load_latest:
        model_path = find_latest_model(algorithm)
        if not model_path:
            print(f"❌ No episode models found for {algorithm}")
            return False
    
    if model_path:
        try:
            agent.load_model(model_path)
            print(f"Successfully loaded model: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return False
    
    return False


def train_rainbow_dqn(env, episodes=1000, save_interval=100, load_model=None, load_best=False, load_latest=False, enable_display=True):
    print("Training with Rainbow DQN...")
    
    frame_shape = env.observation_space['frames'].shape  
    action_history_length = env.observation_space['action_history'].shape[0]  
    n_actions = env.action_space.n  
    
    agent = RainbowDQNAgent(
        input_shape=frame_shape,
        n_actions=n_actions,
        action_history_length=action_history_length,
        lr=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=32,
        target_update=1000,
        n_atoms=51,
        v_min=-10,
        v_max=10,
        multi_step=3,
        enable_display=enable_display
    )
    
    model_loaded = load_model_if_specified(agent, 'rainbow_dqn', load_model, load_best, load_latest)
    if model_loaded:
        print("Continuing training from loaded model")
    else:
        print("Starting training from scratch")

    os.makedirs('models', exist_ok=True)
    os.makedirs('episodes', exist_ok=True)
    
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"Epsilon: {agent.epsilon:.4f}")
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            env.render()
            
            if episode_steps % 100 == 0:
                print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        episode_info = env.get_episode_info()
        agent.end_episode(episode_info)
        agent.update_visualization(episode_reward, episode)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f'models/rainbow_dqn_best.pth')
            print(f"New best reward: {best_reward:.2f} - Model saved!")
        
        if (episode + 1) % save_interval == 0:
            agent.save_model(f'models/rainbow_dqn_episode_{episode + 1}.pth')
            env.reset(save_movie=True, episode=episode)
            print(f"Checkpoint saved at episode {episode + 1}")
    
    print(f"Training completed! Best reward: {best_reward:.2f}")
    return agent


def train_ppo(env, episodes=1000, save_interval=100, load_model=None, load_best=False, load_latest=False, enable_display=True, use_augmentation=True, num_augmentations=2):
    print("Training with PPO...")
    
    if use_augmentation:
        print("Using DrQv2 image augmentations")
    
    binary_mode = hasattr(env, 'binary_mode') and env.binary_mode
    
    if binary_mode:
        frame_shape = env.observation_space.shape 
        n_actions = 6  # [UP, DOWN, LEFT, RIGHT, X, A]
        print(f"Using binary mode with {n_actions} binary actions")
    else:
        frame_shape = env.observation_space.shape 
        n_actions = env.action_space.n
        print(f"Using discrete mode with {n_actions} discrete actions")
    
    agent = PPOAgent(
        input_shape=frame_shape,
        n_actions=n_actions,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_timestep=2048,
        gae_lambda=0.95,
        binary_mode=binary_mode,
        enable_display=enable_display,
        use_augmentation=use_augmentation,
        num_augmentations=num_augmentations
    )
    
    model_loaded = load_model_if_specified(agent, 'ppo', load_model, load_best, load_latest)
    if model_loaded:
        print("Continuing training from loaded model")
    else:
        print("Starting training from scratch")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('episodes', exist_ok=True)
    
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while not done:
            current_binary_input = None
            if hasattr(env, 'binary_mode') and env.binary_mode and hasattr(env, 'inputs'):
                current_binary_input = env.inputs.get_current_binary_input()
            
            action, log_prob, value = agent.act(state, training=True, current_binary_input=current_binary_input)
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, done, log_prob, value)
            agent.update()

            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            env.render()
            
            if episode_steps % 100 == 0:
                print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        episode_info = env.get_episode_info()

        agent.end_episode(episode_info)
        agent.update_visualization(episode_reward, episode)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f'models/ppo_best.pth')
            print(f"New best reward: {best_reward:.2f} - Model saved!")
        
        if (episode + 1) % save_interval == 0:
            agent.save_model(f'models/ppo_episode_{episode + 1}.pth')
            env.reset(save_movie=True, episode=episode)
            print(f"Checkpoint saved at episode {episode + 1}")
    
    print(f"Training completed! Best reward: {best_reward:.2f}")
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train or test RL agents on Mario DS')
    parser.add_argument('--algorithm', type=str, choices=['rainbow', 'ppo'], default='ppo',
                       help='Choose RL algorithm: rainbow (Rainbow DQN) or ppo (PPO)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train/test')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save model every N episodes')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model for testing')
    parser.add_argument('--frame_skip', type=int, default=8,
                       help='Number of frames to skip')
    parser.add_argument('--frame_stack', type=int, default=3,
                       help='Number of frames to stack')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to specific model to load')
    parser.add_argument('--load-best', action='store_true',
                       help='Load the best model for the selected algorithm')
    parser.add_argument('--load-latest', action='store_true',
                       help='Load the latest episode model for the selected algorithm')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable training metrics display window (keep emulator game window)')
    parser.add_argument('--no-game-display', action='store_true',
                       help='Disable emulator game window (completely headless)')
    parser.add_argument('--action-frequency', type=int, default=20,
                       help='Action frequency in Hz - controls how often the agent acts and environment steps (default: 20)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable DrQv2 image augmentations for PPO (enabled by default)')
    parser.add_argument('--num-augmentations', type=int, default=2,
                       help='Number of augmented views per image for DrQv2 (default: 2)')
    
    args = parser.parse_args()
    
    print("Initializing Mario DS Environment...")
    
    ppo_optimized = (args.algorithm == 'ppo')
    binary_mode = True  # Enable binary mode by default
    
    BASE_FREQUENCY = 60
    action_repeat_frames = BASE_FREQUENCY // args.action_frequency
    
    if args.action_frequency > BASE_FREQUENCY:
        print(f"Warning: Action frequency ({args.action_frequency}Hz) is higher than base frequency ({BASE_FREQUENCY}Hz)")
        print("Setting action frequency to match base frequency")
        args.action_frequency = BASE_FREQUENCY
        action_repeat_frames = 1
    
    print(f"Action frequency: {args.action_frequency}Hz")
    print(f"Action repeat frames: {action_repeat_frames}")
    
    if args.no_display and args.no_game_display:
        print("Running in completely headless mode (no displays)")
    elif args.no_display:
        print("Running with emulator display only (no training metrics display)")
    elif args.no_game_display:
        print("Running with training metrics display only (no emulator display)")

    env = MarioDSEnv(
        frame_skip=args.frame_skip, 
        frame_stack=args.frame_stack, 
        ppo_optimized=ppo_optimized, 
        binary_mode=binary_mode, 
        enable_display=not args.no_game_display,
        action_repeat_frames=action_repeat_frames
    )

    try:
        if args.mode == 'train':
            if args.algorithm == 'rainbow':
                train_rainbow_dqn(env, args.episodes, args.save_interval, 
                                        args.load_model, args.load_best, args.load_latest, enable_display=not args.no_display)
            elif args.algorithm == 'ppo':
                train_ppo(env, args.episodes, args.save_interval,
                                args.load_model, args.load_best, args.load_latest, 
                                enable_display=not args.no_display,
                                use_augmentation=not args.no_augmentation,
                                num_augmentations=args.num_augmentations)
        
            if args.model_path is None:
                args.model_path = f'models/{args.algorithm}_best.pth'
            
            if not os.path.exists(args.model_path):
                print(f"Model file not found: {args.model_path}")
                print("Please train a model first or provide a valid model path.")
                return
            
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main()
