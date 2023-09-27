import tensorflow as tf
import numpy as np
import random

class DQN:
    def __init__(self, n_states, n_actions, model_structure=None, trained_model_path='model/', log_dir='logs/', alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = model_structure # This should be just the tensorflow sequential variable filled out
        self.model_path = trained_model_path
        self.log_dir = log_dir
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.file_writer = tf.summary.create_file_writer(self.log_dir)

        # If they gave a path to the model, it will attempt to load it, otherwise it will make the new one
        try:
            self.model = tf.keras.models.load_model(trained_model_path)
        except Exception as e:
            print(f"Couldn't load model because: {e}")
            model_structure.compile(optimizer='adam', loss='mse')  # Typo fixed here
            self.model = model_structure

    def _predict_single_state(self, state):
        return self.model.predict(state, verbose=0)  # Updated shape

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            q_values = self.model.predict(state, verbose=0)  # Updated shape
            return np.argmax(q_values)
    
    # This will learn with the given actions and states, and will return the Q-values
    def learn(self, state, action, reward, next_state, total_reward,  episode):
        target = np.array(reward+self.gamma * np.max(self.model.predict(next_state, verbose=0)))

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            predicted = self.model(state)[0][action]
            predicted = tf.reshape(predicted, (1,)) # Ensure its a 1D tensor
            loss = tf.keras.losses.mean_squared_error(target, predicted)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        with self.file_writer.as_default():
            tf.summary.scalar('Reward', total_reward, step=episode)
            tf.summary.flush()

        return target

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def save(self, path=''):
        self.model.save(path)
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon



# This will build upon the base DQN by helping with overestimation of Q-values
class DoubleDQN:
    def __init__(self, n_states, n_actions, model_structure=None, trained_model_path='model/', log_dir='logs/', alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = model_structure # This should be just the tensorflow sequential variable filled out
        self.model_path = trained_model_path
        self.log_dir = log_dir
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.file_writer = tf.summary.create_file_writer(self.log_dir)

        # If they gave a path to the model, it will attempt to load it, otherwise it will make the new one
        try:
            self.model = tf.keras.models.load_model(trained_model_path)
        except Exception as e:
            print(f"Couldn't load model because {e}")
            model_structure.compile(optimizer='adam', loss='mse')  # Typo fixed here
            self.model = model_structure # Online model
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _predict_single_state(self, state):
        return self.model.predict(state, verbose=0)  # Updated shape


    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            q_values = self._predict_single_state(state)
            return np.argmax(q_values)
    
    # This will learn with the given actions and states, and will return the Q-values
    def learn(self, state, action, reward, next_state):
        # Use the Online Network to find the best action for the next state
        best_action = np.argmax(self.model.predict(next_state, verbose=0))

        # Use the Target Network to find the Q-value of the chosen action
        target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][best_action]
        
        # Existing code to update Online Network
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            predicted = self.model(state)[0][action]
            predicted = tf.reshape(predicted, (1,))
            loss = tf.keras.losses.mean_squared_error(target, predicted)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        with self.file_writer.as_default():
            tf.summary.scalar('Reward', total_reward, step=episode)
            tf.summary.flush()

        return target

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def save(self, path=''):
        self.model.save(path)

    def hard_update(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def soft_update(self, tau=0.005):
        online_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        # Make sure both sets of weights have the same length
        assert len(online_weights) == len(target_weights), "Mismatched weights length!"

        new_weights = []
        for online, target in zip(online_weights, target_weights):
            new_weights.append(tau * online + (1 - tau) * target)

        self.target_model.set_weights(new_weights)
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon



