import numpy as np
import tensorflow as tf
import gym
import random  # Added for random sampling

# Initialize
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
memory = []  # Replay buffer

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Hyperparameters
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate

# Training
for episode in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)
        else:
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward  # Keep track of the total reward
        
        memory.append((state, action, reward, next_state, done))  # Store experience
        
        state = next_state
    
    # Train using replay buffer (randomly sampled)
    batch = random.sample(memory, min(len(memory), 32))
    for state, action, reward, next_state, done in batch:
        target = reward + gamma * np.max(model.predict(next_state, verbose=0)[0]) * (1 - done)
        q_values = model.predict(state, verbose=0)
        q_values[0][action] = target
        model.fit(state, q_values, verbose=0)

    print(f"Episode: {episode}, Total Reward: {total_reward}")  # Print the progress
        
    if epsilon > 0.1:  # Decay exploration rate
        epsilon *= 0.995
