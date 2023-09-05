import numpy as np
import tensorflow as tf
from random import randrange

class MarioDQN:
    def __init__(self, n_states, n_actions, TOTAL_PIXELS, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.TOTAL_PIXELS = TOTAL_PIXELS
        try:
            self.model = tf.keras.models.load_model('models/model.h5')
        except:
            self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.TOTAL_PIXELS,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def choose_action(self, frame_stack):
        if np.random.rand() < self.epsilon:
            return randrange(self.n_actions)
        else:
            return self.model.predict(frame_stack.reshape(1, -1), verbose=0)[0]

    def learn(self, frame_stack, action, reward, next_frame_stack):
        target = reward + self.gamma * np.max(self.model.predict(next_frame_stack.reshape(1, -1), verbose=0))
        target = np.array([target])  # Ensure it's a 1D tensor
        
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            predicted = self.model(frame_stack.reshape(1, -1))[0][action]
            predicted = tf.reshape(predicted, (1,))  # Ensure it's a 1D tensor
            loss = tf.keras.losses.mean_squared_error(target, predicted)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, frame_stack):
        stacked_frames_array = np.array(frame_stack).reshape(-1)  # Flatten the deque into a single numpy array
        return self.model.predict(stacked_frames_array.reshape(1, -1), verbose=0)

    def save(self):
        self.model.save('models/model')
