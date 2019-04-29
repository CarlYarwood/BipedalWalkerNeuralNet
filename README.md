# BipedalWalkerNeuralNetworks
Project for Artificial Intelligence class at Truman State University. Compares performance of a neural network trained using a genetic algorithm and a reinforcement learning algorithm, meant to perform in OpenAI's CartPole gym environment.

## Dependencies
- [Tensorflow](https://www.tensorflow.org/install/)
- [Keras](https://keras.io/#installation)
- [OpenAI's Gym](https://gym.openai.com/docs/)
- [h5py (for loading/storing Keras models)](https://keras.io/getting-started/faq/#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras)

## Running the code
To run the DQN model you have a few options: if you want to run just the best model, with no training, execute "python3 DQN_Runner.py -t". If you want to see  it train, leave off the -t option. If you'd like to save the best weights, add -s. If you want it to render after every best episode, add -s and -r.

To run the genetic algorithm, just run "python3 GeneticAlgorithm.py"