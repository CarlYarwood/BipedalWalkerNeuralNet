from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from collections import deque
import numpy as np
import random

# TODO Documentation

class DQN():
    """
    Implements a deep Q learning network, based on the paper released by Google:
    Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G.,
        . . . Hassabis, D. (2015). Human-level control through deep reinforcement learning.
        Nature, 518(7540), 529-533. doi:10.1038/nature14236 
    """

    
    def __init__(self, env, learning_rate=0.00125, gamma=0.99, batch_size=32,
                 max_replays=1000000, starting_epsilon=1.0, epsilon_decay=0.99,
                 min_epsilon=0.01, target_model_update_freq=500, load_weights=False):
        """
        Initialize variables.
        @param env: environment from OpenAI's gym.
        @param learning_rate: learning rate for the network
        @param gamma: factor for how much we discount future reward
        @param batch_size: how many time steps we train on when replaying
        @param max_replays: maximum number of replays to store
        @param starting_epsilon: epsilon is the chance we take a random action.
                starting_epsilon is the value we begin with
        @param epsilon_decay: percent epsilon decays each episode
        @param min_epsilon: minimum value for epsilon. Will not decay below this
        @param target_model_update_freq: we copy our current model to the target
                model once every this many timesteps
        """
        self.env = env
        self.state_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.learning_rate = learning_rate
        self.gamma = gamma # Discount factor for future reward
        self.epsilon = starting_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_model_update_freq = target_model_update_freq
        self.total_timesteps = 0
        self.best_score = 0
        self.best_model = None
    
        self.memory = deque(maxlen=max_replays)

        self.model = self.build_model()
        self.target_model = self.build_model()

        if load_weights:
            self.load_weights()

    def build_model(self):
        """ 
        Build the neural network using Keras.
        Used for the target network as well as the acting network.
        """

        model = Sequential()
        model.add(Dense(64, input_shape=self.state_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        
        model.compile(loss='mse',
                    optimizer=optimizers.Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        """ 
        Take an action.
        With probability epsilon, take a random action. Otherwise,
        take the best action as predicted by the network.
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def remember(self, state, action, next_state, reward, done):
        """
        Remember a timestep to later train on it. The deque structure
        has a maximum number of timesteps it will store and then
        older ones will be removed.
        The target network's weights will also be updated here, if it
        is necessary.
        parameters are self-explanatory
        """

        self.memory.append([state, action, next_state, reward, done]) # TODO remove done. or maybe not idk
        self.total_timesteps += 1
        if self.total_timesteps % self.target_model_update_freq == 0:
            print("Updating target network!")
            self.target_model.set_weights(self.model.get_weights()) 

    def replay(self):
        """
        Replay a batch of memories and train the network on it.
        """
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        states = []
        targets = []

        for state, action, next_state, reward, done in batch:
            if done:
                discounted_reward = reward
            else:
                discounted_reward = reward + self.gamma * np.amax(self.model.predict(next_state))

            target = self.target_model.predict(state)
            target[0][action] = discounted_reward

            states.append(state.flatten())
            targets.append(target.flatten())

        self.model.fit(np.array(states), np.array(targets), batch_size=len(batch), verbose=0)

    def decay_epsilon(self):
        """
        Decay the epsilon value.
        """
        if self.epsilon > self.min_epsilon: self.epsilon *= self.epsilon_decay

    def maybe_save_model(self, score):
        """
        Save the model if it has obtained the best score so far.
        @param score: score for this episode
        @returns: true if model was saved
        """
        if score >= self.best_score:
            self.best_score = score
            self.model.save_weights("rl_model.h5")
            return True

        return False

    def load_weights(self):
        self.model.load_weights("rl_model.h5")