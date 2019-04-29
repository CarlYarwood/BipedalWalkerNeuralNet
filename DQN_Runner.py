#!/usr/bin/python3
 
import argparse
import gym
import numpy as np
from DQN import DQN

NUM_EPISODES = 400
MAX_TIMESTEPS = 200
NUM_TRIALS = 10
NUM_RANDOM_EPISODES = 100

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", help="if true, load the best weights", action="store_true")
    parser.add_argument("-r", "--render", help="if true, render episode after each best-so-far model", action="store_true")
    parser.add_argument("-s", "--save_weights", help="save the weights of the best model. Must not be in test mode",
                        action="store_true")
    args = parser.parse_args()

    if args.save_weights and args.test:
        print("Error, will not save weights if in testing mode!")
        quit()

    render = False
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    state = env.reset().reshape(1, state_size)


    if not args.test:
        scores = []
        scores_file = open("dqn_scores.csv", 'w')

        for episode in range(NUM_EPISODES):
            scores_file.write(f"Episode {episode}, ")

        scores_file.write("\n")

    # run a number of trials so we can get an average of performance
    for trial_num in range(NUM_TRIALS):
        training = False
        state = env.reset().reshape(1, state_size)
        model = DQN(env, load_weights=args.test, starting_epsilon=0.0 if args.test else 1.0)
        for e in range(NUM_EPISODES + NUM_RANDOM_EPISODES):
            if e > NUM_RANDOM_EPISODES:
                training = not args.test

            state = env.reset().reshape(1, state_size)
            for t in range(MAX_TIMESTEPS):
                if render or args.test: env.render()
                action = model.act(state)

                old_state = state
                state, reward, done, _ = env.step(action)
                state = state.reshape(1, state_size)
                if not args.test:
                    model.remember(old_state, action, state, reward, done)
                
                if training:
                    model.replay()

                if done:
                    break
            
            print(f"Episode {e} lasted {t} frames. epsilon: {model.epsilon}. Total time steps: {model.total_timesteps}")

            if training:
                render = False
                if args.save_weights:
                    if model.maybe_save_model(t):
                        print("Best so far, saving model!")
                        if args.render:
                            render = True
                            
                model.decay_epsilon()
                scores.append(t)

        if not args.test:
            scores_file.write(f"Trial {trial_num}, ")
            for score in scores:
                scores_file.write(f"{score},")

            scores_file.write("\n")
            scores_file.flush()

            scores = []

    env.close()
    if not args.test:
        scores_file.close()

if __name__ == "__main__":
    main()