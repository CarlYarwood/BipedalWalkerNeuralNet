import gym
from Model import Model
import numpy as np

NUM_EPISODES = 500
MAX_TIMESTEPS = 200
NUM_TRIALS = 500

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
state = env.reset().reshape(1, state_size)

model = Model(env)
scores = []
scores_file = open("rl_scores.csv", 'w')

for episode in range(NUM_EPISODES):
    scores_file.write(f"Episode {episode}, ")

scores_file.write("\n")

for trial_num in range(NUM_TRIALS):
    for e in range(NUM_EPISODES):
        state = env.reset().reshape(1, state_size)
        for t in range(MAX_TIMESTEPS):
            # if e % 50 == 0: env.render()
            action = model.act(state)

            old_state = state
            state, reward, done, info = env.step(action)
            state = state.reshape(1, state_size)
            model.remember(old_state, action, state, reward, done)
            
            if done:
                break
            model.replay()
        model.maybe_save_model(t)
        model.decay_epsilon()
        scores.append(t)
        print(f"Episode {e} lasted {t} frames. epsilon: {model.epsilon}.")

    scores_file.write(f"Trial {trial_num}, ")
    for score in scores:
        scores_file.write(f"{score},")

    scores_file.write("\n")

    scores = []

env.close()
scores_file.close()