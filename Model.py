from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from collections import deque
import numpy as np
import random

class Model():

    
    def __init__(self, state_shape, num_actions, learning_rate=0.01, gamma=0.95, batch_size=32, max_replays=2000, starting_epsilon=1.0):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma # Reward decay
        self.epsilon = starting_epsilon
        self.batch_size = batch_size
    
        self.memory = deque(maxlen=max_replays)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=self.state_shape, activation='relu'))
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
        self.memory.append([state, action, next_state, reward, done])

    def replay(self):
        if self.epsilon > 0.01: self.epsilon *= 0.995

        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        states = []
        targets = []

        for state, action, next_state, reward, done in batch:
            if done:
                discounted_reward = reward
            else:
                discounted_reward = reward + self.gamma * np.amax(self.model.predict(next_state))

            target = self.model.predict(state)
            target[0][action] = discounted_reward

            states.append(state.flatten())
            targets.append(target.flatten())

        self.model.fit(np.array(states), np.array(targets), batch_size=len(batch), verbose=0)