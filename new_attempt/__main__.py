from DQN import *
import tensorflow as tf
import gym
import ale_py
from ale_py import ALEInterface

ale = ALEInterface()
ale = gym.make('Pong-v0')

n_actions = ale.action_space.n
n_states = ale.observation_space.shape  # This will give you a tuple like (height, width, channels)


def run_experiment(agent, env_name, episodes=1000):
    env = gym.make(env_name)
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            print(state)
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.learn(state, action, reward, next_state, total_reward, episode)
            state = next_state

            env.render()
            
        
        total_rewards.append(total_reward)
    return total_rewards

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),  # Assuming input frames are 84x84 and stack of 4 frames
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(n_actions)  # Linear activation to output raw Q-values
]) 

# Initialize DQN and DoubleDQN agents
dqn_agent = DQN(n_states=(84,84,4), n_actions=n_actions, model_structure=model)
double_dqn_agent = DoubleDQN(n_states=(84,84,4), n_actions=n_actions, model_structure=model)

# Run experiments
dqn_rewards = run_experiment(dqn_agent, 'Pong-v0')
double_dqn_rewards = run_experiment(double_dqn_agent, 'Pong-v0')