from NueralNetwork import NueralNetwork
from Node import Node
from FirstLayerNode import FirstLayerNode
import numpy as np
import gym
import random

class GeneticAlgorithm:

    """
    Initialize population of nueral networks
    """
    def __init__(self):
        self.SPOP = 8 #Population size
        
        self.population = []
        for i in range(self.SPOP):
            # Create empty arrays for each layer and load them into the Nueral Network to
            # be initialized with nodes of random weights and bias
            firstLayer = []
            hiddenLayer = []
            outputLayer = []
            self.population.append(NueralNetwork(firstLayer, hiddenLayer, outputLayer))

        self.selectionPool = []

        self.parentGen = []
        
        
    def evaluateFitness(self):
        env = gym.make('CartPole-v1')
        for nn in range(self.SPOP):
            observation = env.reset()
            for t in range(100):
                env.render()
                action = self.population[nn].makeDecision(observation)
                observation, reward, done, info = env.step(action)
                if done:
                    for i in range(t+1):
                        self.selectionPool.append(self.population[nn])
                    print("Network {} earned fitness score of {}".format(nn, t+1))
                    break
        env.close()
        
        
    def selectMatingPool(self):
        for i in range(int(self.SPOP * 3 / 2)):
            self.parentGen.append(self.selectionPool[random.randint(0, len(self.selectionPool) - 1)])

    #def mutate(self):

    def mate(self, nn1, nn2):
        firstLayer = []
        for i in range(len(nn1.firstLayer)):
            print(nn1.firstLayer[i])
            print(nn2.firstLayer[i])
            firstLayer.append(nn1.combineFirstLayerNodes(nn1.firstLayer[i], nn2.firstLayer[i]))

        hiddenLayer = []
        for i in range(len(nn1.hiddenLayer)):
            n = NueralNetwork.combineNodes(nn1.hiddenLayer[i], nn2.hiddenLayer[i])
            n.setPrevNode(firstLayer)
            hiddenLayer.append(n)

        outputLayer = []
        for i in range(len(nn1.outputLayer)):
            n = NueralNetwork.combineNodes(nn1.outputLayer[i], nn2.outputLayer[i])
            n.setPrevNode(hiddenLayer)
            hiddenLayer.append(n)

        return NueralNetwork(firstLayer, hiddenLayer, outputLayer)
        
    def createNextGen(self):
        self.population = []
        for i in range(0, int(self.SPOP * 3/2), 2):
            self.population.append(self.mate(self.parentGen[i], self.parentGen[i + 1]))
        for i in range(0, int(self.SPOP * 1/4)):
            self.population.append(NueralNetwork())
            
            

ga = GeneticAlgorithm()
ga.evaluateFitness()
ga.selectMatingPool()
ga.createNextGen()
ga.evaluateFitness()
ga.selectMatingPool()
ga.createNextGen()
ga.evaluateFitness()
ga.selectMatingPool()
ga.createNextGen()


