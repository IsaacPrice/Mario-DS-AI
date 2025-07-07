#!/usr/bin/env python3

"""
Simple PPO training script to demonstrate the optimized environment
"""

import numpy as np
import torch
from mario_env import MarioDSEnv
from ppo_agent import PPOAgent
import time

def train_ppo_simple(episodes=10, max_steps=500):
    """
    Simple PPO training to demonstrate the optimized environment
    """
    print("Starting PPO training with optimized environment...")
    
    # Create PPO-optimized environment
    env = MarioDSEnv(frame_skip=4, frame_stack=4, ppo_optimized=True)
    
    # Create PPO agent with smaller update timestep for demo
    agent = PPOAgent(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        update_timestep=128,  # Smaller for faster updates
        gae_lambda=0.95
    )
    
    print(f"Environment: {env.action_space.n} actions, observation shape: {env.observation_space.shape}")
    print(f"Action mapping for PPO:")
    print("  0: Do nothing")
    print("  1: Walk right")
    print("  2: Run right")
    print("  3: Jump")
    print("  4: Jump right")
    print("  5: Walk left")
    
    episode_rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        start_time = time.time()
        
        while steps < max_steps:
            # Select action
            action, log_prob, value = agent.select_action(obs)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.memory.store(obs, action, reward, done, log_prob, value)
            
            # Update agent
            agent.update()
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            steps += 1
            
            # Render environment
            env.render()
            
            # Print progress
            if steps % 100 == 0:
                print(f"  Steps: {steps}, Reward: {episode_reward:.3f}, Action: {action}")
            
            if done:
                print(f"  Episode finished at step {steps} (Mario died)")
                break
        
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Reward: {episode_reward:.3f}")
        print(f"  Steps: {steps}")
        print(f"  Time: {episode_time:.2f}s")
        print(f"  Average reward: {np.mean(episode_rewards):.3f}")
        
        # Update visualization
        agent.update_visualization(episode_reward, episode)
    
    env.close()
    print(f"\nTraining completed!")
    print(f"Average reward over {episodes} episodes: {np.mean(episode_rewards):.3f}")
    print(f"Best episode reward: {max(episode_rewards):.3f}")
    
    return agent, episode_rewards

def compare_environments():
    """
    Compare PPO vs DQN environment performance
    """
    print("\nComparing PPO vs DQN environment setups...")
    
    # Test PPO environment
    print("Testing PPO environment (6 actions, simplified observations)...")
    env_ppo = MarioDSEnv(frame_skip=4, frame_stack=4, ppo_optimized=True)
    
    obs = env_ppo.reset()
    total_reward_ppo = 0
    for i in range(100):
        action = env_ppo.action_space.sample()
        obs, reward, done, truncated, info = env_ppo.step(action)
        total_reward_ppo += reward
        if done:
            break
    
    env_ppo.close()
    
    # Test DQN environment
    print("Testing DQN environment (8 actions, complex observations)...")
    env_dqn = MarioDSEnv(frame_skip=4, frame_stack=4, ppo_optimized=False)
    
    obs = env_dqn.reset()
    total_reward_dqn = 0
    for i in range(100):
        action = env_dqn.action_space.sample()
        obs, reward, done, truncated, info = env_dqn.step(action)
        total_reward_dqn += reward
        if done:
            break
    
    env_dqn.close()
    
    print(f"\nComparison results (100 random steps):")
    print(f"PPO environment reward: {total_reward_ppo:.3f}")
    print(f"DQN environment reward: {total_reward_dqn:.3f}")
    print(f"PPO reward improvement: {((total_reward_ppo - total_reward_dqn) / abs(total_reward_dqn) * 100):.1f}%")

if __name__ == "__main__":
    try:
        # Run simple training
        agent, rewards = train_ppo_simple(episodes=5, max_steps=300)
        
        # Compare environments
        compare_environments()
        
        print("\n✅ PPO environment optimization completed successfully!")
        print("\nKey improvements made:")
        print("1. Simplified observation space (no action history)")
        print("2. Reduced action space (6 vs 8 actions)")
        print("3. Better reward shaping for PPO")
        print("4. Optimized action mapping for Mario gameplay")
        print("5. Faster training with more informative rewards")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
