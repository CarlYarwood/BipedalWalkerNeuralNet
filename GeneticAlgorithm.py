<<<<<<< HEAD
import numpy as np
import random
from NueralNetwork import NueralNetwork
import gym


POPSIZE = 64

def initializePopulation():
    population = []
    for i in range(POPSIZE):
        wih = np.random.random((4,24))
        who = np.random.random((24,2))
        b0 = np.random.random((4,))
        b1 = np.random.random((24,))
        b2 = np.random.random((2,))
        population.append(NueralNetwork(wih,who,b0,b1,b2))
    return population

def evaluateFitness(nn, env):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = nn.makeDecision(observation)
        observation, reward, done, info = env.step(action)
        if done:
            return t + 1
            break
    return 100

def buildSelectionPool(population, threshhold):
    selectionPool = []
    th = threshhold
    env = gym.make('CartPole-v1')
    for i in range(len(population)):
        fitness = evaluateFitness(population[i], env)
        print("Network {} earned fitness score of {}".format(i, fitness))
        if (fitness > th):
            for j in range(fitness - th):
                selectionPool.append(population[i])
    env.close()
    return selectionPool

def mate(nn1, nn2):
    """
    if (random.randint(0,1) == 0):
        wih = nn1.wih
    else:
        wih = nn2.wih
    for i in range(wih.shape[0]):
        for j in range(wih.shape[1]):
            if (random.randint(0, 10) == 0):
                wih[i][j] = random.random()

    if (random.randint(0,1) == 0):
        who = nn1.who
    else:
        who = nn2.who
    for i in range(who.shape[0]):
        for j in range(who.shape[1]):
            if (random.randint(0, 10) == 0):
                who[i][j] = random.random()

    if (random.randint(0,1) == 0):
        b0 = nn1.b0
    else:
        b0 = nn2.b0
    for i in range(b0.size):
        if (random.randint(0, 10) == 0):
            b0[i] = random.random()

    if (random.randint(0,1) == 0):
        b1 = nn1.b1
    else:
        b1 = nn2.b1
    for i in range(b1.size):
        if (random.randint(0, 10) == 0):
            b1[i] = random.random()

    if (random.randint(0,1) == 0):
        b2 = nn2.b2
    else:
        b2 = nn2.b2
    for i in range(b2.size):
        if (random.randint(0, 10) == 0):
            b2[i] = random.random()
    
    """
    wih = np.empty(nn1.wih.shape)
    for i in range(nn1.wih.shape[0]):
        for j in range(nn1.wih.shape[1]):
            if (random.randint(0, 1) == 0):
                wih[i][j] = nn1.wih[i][j]
            else:
                wih[i][j] = nn2.wih[i][j]
    
    who = np.empty(nn1.who.shape)
    for i in range(nn1.who.shape[0]):
        for j in range(nn1.who.shape[1]):
            if (random.randint(0, 1) == 0):
                who[i][j] = nn1.who[i][j]
            else:
                who[i][j] = nn2.who[i][j]
    b0 = np.empty((nn1.b0.size,))
    for i in range(nn1.b0.size):
        if (random.randint(0, 1) == 0):
            b0[i] = nn1.b0[i]
        else:
            b0[i] = nn2.b0[i]

    b1 = np.empty((nn1.b1.size,))
    for i in range(nn1.b1.size):
        if (random.randint(0, 1) == 0):
            b1[i] = nn1.b1[i]
        else:
            b1[i] = nn2.b1[i]

    b2 = np.empty((nn1.b2.size,))
    for i in range(nn1.b2.size):
        if (random.randint(0, 1) == 0):
            b2[i] = nn1.b2[i]
        else:
            b2[i] = nn2.b2[i]
    

    return NueralNetwork(wih, who, b0, b1, b2)
        


def buildNextGen(selectionPool):
    population = []
    for i in range(int(POPSIZE * 7/8)):
        x = random.randint(0, len(selectionPool) - 1)
        y = random.randint(0, len(selectionPool) - 1)
        population.append(mate(selectionPool[x], selectionPool[y]))
    for i in range(int(POPSIZE * 1/8)):
        wih = np.random.random((4,24))
        who = np.random.random((24,2))
        b0 = np.random.random((4,))
        b1 = np.random.random((24,))
        b2 = np.random.random((2,))
        population.append(NueralNetwork(wih,who,b0,b1,b2))
    return population

def train():
    pop = initializePopulation()
    threshhold = 9
    for i in range(100):
        print()
        print()
        print("Evaluating Gen {}".format(i))
        print()
        print()
        pool = buildSelectionPool(pop, threshhold)
        pop = buildNextGen(pool)

train()
        
=======
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


>>>>>>> c95a6d2cb4bfac41cbed048bf7db4466da2b162e
