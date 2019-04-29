from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from collections import deque
import numpy as np
import random

# TODO Documentation

class Model():

    
    def __init__(self, env, learning_rate=0.00025, gamma=0.99, batch_size=32,
                 max_replays=1000000, starting_epsilon=1.0, epsilon_decay=0.99,
                 min_epsilon=0.01):
        self.env = env
        self.state_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.learning_rate = learning_rate
        self.gamma = gamma # Discount factor for future reward
        self.epsilon = starting_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.best_score = 0
        self.best_model = None
    
        self.memory = deque(maxlen=max_replays)

        self.model = Sequential()
        self.model.add(Dense(64, input_shape=self.state_shape, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='linear'))
        
        self.model.compile(loss='mse',
                    optimizer=optimizers.Adam(lr=self.learning_rate))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        return np.argmax(self.model.predict(state))

    def remember(self, state, action, next_state, reward, done):
        self.memory.append([state, action, next_state, reward, done]) # TODO remove done

    def replay(self):
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        states = []
        targets = []

        for state, action, next_state, reward, done in batch:
            # reward = reward if not done else -10
            if done:
                discounted_reward = reward
            else:
                discounted_reward = reward + self.gamma * np.amax(self.model.predict(next_state))

            target = self.model.predict(state)
            target[0][action] = discounted_reward

            states.append(state.flatten())
            targets.append(target.flatten())

        self.model.fit(np.array(states), np.array(targets), batch_size=len(batch), verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

    def maybe_save_model(self, score):
        if score > self.best_score:
            self.best_score = score
            self.model.save_weights("rl_model.h5")
            print("Best so far, saving model!")