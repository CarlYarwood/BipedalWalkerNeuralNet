from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import random

class Model():

    
    def __init__(self, state_shape, num_actions, learning_rate=0.1, gamma=0.8, batch_size=50):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma # Reward decay
        self.batch_size = batch_size
    
        self.memory = []

        self.model = Sequential()
        self.model.add(Dense(32, input_shape=self.state_shape, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='linear'))
        
        self.model.compile(loss='mse',
                    optimizer=optimizers.Adam(lr=self.learning_rate))

    # TODO exploration vs exploitation
    def predict(self, state):
        return self.model.predict(state.reshape(1, self.state_shape[0])).flatten()

    def remember(self, state, action, next_state, reward, done):
        self.memory.append([state, action, next_state, reward, done])

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)

        for state, action, next_state, reward, done in batch:
            if done:
                target = reward
            else:
                target = reward # TODO im confuse