import gym
import numpy as np
import tensorflow as tf
from DQN import *  # Import your DQN class
import pygame
from gym.utils.play import play

# Initialize environment and DQN agent
env = gym.make("CarPole-v4", render_mode='human')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
epsilon = 1

# Create your model structure before passing it to DQN
model_structure = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(n_actions, activation='linear')
])

agent = DoubleDQN(n_states, n_actions, epsilon=epsilon, model_structure=model_structure)

# Training loop
n_episodes = 1000
for episode in range(n_episodes):
    state, _ = env.reset()
    state = np.array(state)
    state = np.reshape(state, [1, n_states])
    total_reward = 0

    while True:
        # Uncomment this if you want to see the game
        env.render()

        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, n_states])

        # Update total reward
        total_reward += reward

        # Learn from experience
        agent.learn(state, action, reward, next_state, total_reward, episode)

        state = next_state

        if done:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")
            break
    
    agent.soft_update(.2)
    if epsilon > .0001:
        epsilon *= .995
        agent.set_epsilon(epsilon)


    # Save the model every 100 episodes
    if episode % 100 == 0:
        agent.save('path_to_save_model/')

# Close the environment
env.close()
