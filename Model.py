from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from collections import deque
import numpy as np
import random

class Model():

    
    def __init__(self, state_shape, num_actions, learning_rate=0.001, gamma=0.8, batch_size=50, max_replays=2000, starting_epsilon=1.0):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma # Reward decay
        self.epsilon = starting_epsilon
        self.batch_size = batch_size
        self.max_replays = max_replays

    
        self.memory = deque()

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=self.state_shape, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='linear'))
        
        self.model.compile(loss='mse',
                    optimizer=optimizers.Adam(lr=self.learning_rate))

    # TODO exploration vs exploitation
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        return np.argmax(self.model.predict(state))

    def remember(self, state, action, next_state, reward, done):
        if len(self.memory) > self.max_replays: self.memory.popleft()
        self.memory.append([state, action, next_state, reward, done])

    def replay(self):
        if self.epsilon > 0.01: self.epsilon *= 0.995
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = list(self.memory)
            random.shuffle(batch)

        for state, action, next_state, reward, done in batch:
            if done:
                discounted_reward = reward
            else:
                discounted_reward = reward + self.gamma * np.amax(self.model.predict(next_state))

            target = self.model.predict(state)
            target[0][action] = discounted_reward

            self.model.fit(state, target, verbose=0)