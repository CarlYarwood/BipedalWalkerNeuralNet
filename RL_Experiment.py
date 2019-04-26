import gym
from Model import Model

NUM_EPISODES = 100

env = gym.make('BipedalWalker-v2')
state = env.reset()

model = Model(state_shape = env.observation_space.shape, num_actions = env.action_space.shape[0])

for e in range(NUM_EPISODES):
    for _ in range(500):
        env.render()
        action = model.predict(state)
        old_state = state
        state, reward, done, info = env.step(action)
        model.remember(old_state, action, state, reward, done)

        if done:
            break
        # env.step(env.action_space.sample()) # take a random action
env.close()