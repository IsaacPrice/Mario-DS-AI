#!/usr/bin/env python3

"""
Test script to verify PPO-optimized environment works correctly
"""

import numpy as np
from mario_env import MarioDSEnv
from ppo_agent import PPOAgent
import torch

def test_ppo_environment():
    """Test that the PPO environment works properly"""
    print("Testing PPO-optimized environment...")
    
    # Create PPO-optimized environment
    env = MarioDSEnv(frame_skip=4, frame_stack=4, ppo_optimized=True)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test environment reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}, Obs shape={obs.shape}")
        
        if done:
            print("Episode finished, resetting...")
            obs = env.reset()
    
    env.close()
    print("PPO environment test completed successfully!")

def test_ppo_agent():
    """Test that the PPO agent works with the environment"""
    print("\nTesting PPO agent...")
    
    # Create PPO-optimized environment
    env = MarioDSEnv(frame_skip=4, frame_stack=4, ppo_optimized=True)
    
    # Create PPO agent
    agent = PPOAgent(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        lr=3e-4,
        update_timestep=32  # Small for testing
    )
    
    print(f"Agent created with {env.action_space.n} actions")
    
    # Test agent interaction
    obs = env.reset()
    total_reward = 0
    
    for i in range(50):
        action, log_prob, value = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Store transition
        agent.memory.store(obs, action, reward, done, log_prob, value)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}: Action={action}, Reward={reward:.3f}, Value={value:.3f}")
        
        if done:
            print(f"Episode finished at step {i}")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Memory size: {len(agent.memory.states)}")
    
    env.close()
    print("PPO agent test completed successfully!")

def test_dqn_environment():
    """Test that DQN environment still works"""
    print("\nTesting DQN environment compatibility...")
    
    # Create DQN-style environment
    env = MarioDSEnv(frame_skip=4, frame_stack=4, ppo_optimized=False)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test environment reset
    obs = env.reset()
    print(f"Initial observation type: {type(obs)}")
    print(f"Frames shape: {obs['frames'].shape}")
    print(f"Action history shape: {obs['action_history'].shape}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={done}")
        
        if done:
            print("Episode finished, resetting...")
            obs = env.reset()
    
    env.close()
    print("DQN environment test completed successfully!")

if __name__ == "__main__":
    try:
        test_ppo_environment()
        test_ppo_agent()
        test_dqn_environment()
        print("\n✅ All tests passed! Environment is ready for PPO training.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
