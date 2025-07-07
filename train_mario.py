import argparse
import numpy as np
import time
import os
from mario_env import MarioDSEnv
from rainbow_dqn import RainbowDQNAgent
from ppo_agent import PPOAgent
import matplotlib.pyplot as plt
import torch

def train_rainbow_dqn(env, episodes=1000, save_interval=100):
    """Train Rainbow DQN agent"""
    print("Training with Rainbow DQN...")
    
    # Initialize agent
    frame_shape = env.observation_space['frames'].shape  # (4, 48, 64)
    action_history_length = env.observation_space['action_history'].shape[0]  # 10
    n_actions = env.action_space.n  # 8 actions
    
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
        multi_step=3
    )
    
    # Create directories for saving
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
            # Select action
            action = agent.act(state, training=True)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Render environment
            env.render()
            
            if episode_steps % 100 == 0:
                print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        # Update visualization
        agent.update_visualization(episode_reward, episode)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f'models/rainbow_dqn_best.pth')
            print(f"New best reward: {best_reward:.2f} - Model saved!")
        
        # Save periodic checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save_model(f'models/rainbow_dqn_episode_{episode + 1}.pth')
            # Save episode video
            env.reset(save_movie=True, episode=episode)
            print(f"Checkpoint saved at episode {episode + 1}")
    
    print(f"Training completed! Best reward: {best_reward:.2f}")
    return agent

def train_ppo(env, episodes=1000, save_interval=100):
    """Train PPO agent"""
    print("Training with PPO...")
    
    # Initialize agent
    frame_shape = env.observation_space.shape  # (4, 48, 64) - simplified for PPO
    n_actions = env.action_space.n  # 10 actions for enhanced PPO
    
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
        gae_lambda=0.95
    )
    
    # Create directories for saving
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
            # Select action
            action, log_prob, value = agent.act(state, training=True)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, done, log_prob, value)
            
            # Update policy if enough data collected
            agent.update()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Render environment
            env.render()
            
            if episode_steps % 100 == 0:
                print(f"  Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        
        # Update visualization
        agent.update_visualization(episode_reward, episode)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f'models/ppo_best.pth')
            print(f"New best reward: {best_reward:.2f} - Model saved!")
        
        # Save periodic checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save_model(f'models/ppo_episode_{episode + 1}.pth')
            # Save episode video
            env.reset(save_movie=True, episode=episode)
            print(f"Checkpoint saved at episode {episode + 1}")
    
    print(f"Training completed! Best reward: {best_reward:.2f}")
    return agent

def test_agent(env, agent_type, model_path, episodes=5):
    """Test a trained agent"""
    print(f"Testing {agent_type} agent...")
    
    if agent_type.lower() == 'rainbow':
        frame_shape = env.observation_space['frames'].shape
        action_history_length = env.observation_space['action_history'].shape[0]
        n_actions = env.action_space.n
        agent = RainbowDQNAgent(frame_shape, n_actions, action_history_length)
    elif agent_type.lower() == 'ppo':
        frame_shape = env.observation_space.shape
        n_actions = env.action_space.n
        agent = PPOAgent(frame_shape, n_actions)
    else:
        raise ValueError("Agent type must be 'rainbow' or 'ppo'")
    
    # Load trained model
    agent.load_model(model_path)
    
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nTest Episode {episode + 1}/{episodes}")
        
        while not done:
            # Select action (no training)
            if agent_type.lower() == 'rainbow':
                action = agent.act(state, training=False)
            else:  # PPO
                action = agent.act(state, training=False)
            
            # Take action
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            time.sleep(0.02)  # Slow down for viewing
        
        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1} - Reward: {episode_reward:.2f}, Steps: {steps}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\nTest Results - Average Reward: {avg_reward:.2f}")
    
    return total_rewards

def main():
    parser = argparse.ArgumentParser(description='Train or test RL agents on Mario DS')
    parser.add_argument('--algorithm', type=str, choices=['rainbow', 'ppo'], default='rainbow',
                       help='Choose RL algorithm: rainbow (Rainbow DQN) or ppo (PPO)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train/test')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='Save model every N episodes')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model for testing')
    parser.add_argument('--frame_skip', type=int, default=4,
                       help='Number of frames to skip')
    parser.add_argument('--frame_stack', type=int, default=4,
                       help='Number of frames to stack')
    
    args = parser.parse_args()
    
    # Initialize environment
    print("Initializing Mario DS Environment...")
    
    # Use PPO-optimized environment for PPO, DQN-style for Rainbow
    ppo_optimized = (args.algorithm == 'ppo')
    env = MarioDSEnv(frame_skip=args.frame_skip, frame_stack=args.frame_stack, ppo_optimized=ppo_optimized)
    
    try:
        if args.mode == 'train':
            if args.algorithm == 'rainbow':
                agent = train_rainbow_dqn(env, args.episodes, args.save_interval)
            elif args.algorithm == 'ppo':
                agent = train_ppo(env, args.episodes, args.save_interval)
        
        elif args.mode == 'test':
            if args.model_path is None:
                # Use default best model path
                args.model_path = f'models/{args.algorithm}_best.pth'
            
            if not os.path.exists(args.model_path):
                print(f"Model file not found: {args.model_path}")
                print("Please train a model first or provide a valid model path.")
                return
            
            test_agent(env, args.algorithm, args.model_path, args.episodes)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main()
