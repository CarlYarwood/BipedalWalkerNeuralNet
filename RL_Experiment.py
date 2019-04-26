import gym
from Model import Model

NUM_EPISODES = 1000

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
state = env.reset().reshape(1, state_size)

model = Model(state_shape = env.observation_space.shape, num_actions = env.action_space.n)

for e in range(NUM_EPISODES):
    state = env.reset().reshape(1, state_size)
    for t in range(500):
        if e > 200 and e % 5 == 0: env.render()
        action = model.act(state)
        old_state = state
        state, reward, done, info = env.step(action)
        state = state.reshape(1, state_size)
        model.remember(old_state, action, state, reward, done)

        if done:
            break
        # env.step(env.action_space.sample()) # take a random action
    model.replay()
    print(f"Episode {e} lasted {t} frames. epsilon: {model.epsilon}.")
env.close()