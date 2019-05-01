import numpy as np
import random
from NueralNetwork import NueralNetwork
import gym


POPSIZE = 64
MAX_TIMESTEPS = 200
MAX_GEN = 200
TRIALS = 4 # number of trials
MR = 100 # Mutation Rate (MR / 100)%
SEED = 3


L0 = 4 # number of input nodes
L1 = 24 # number first hidden layer nodes
L2 = 2  # number output nodes


np.random.seed(SEED)

"""
Randomly initialize weights and biases for a population of nueral networks

@return array of randomly generated nueral networks
"""
def initializePopulation():
    population = []
    for i in range(POPSIZE):
        wih = np.random.random((L0, L1))
        who = np.random.random((L1, L2))
        b0 = np.random.random((L0,))
        b1 = np.random.random((L1,))
        b2 = np.random.random((L2,))
        population.append(NueralNetwork(wih,who,b0,b1,b2))
    return population


"""
Excecute run of a Nueral Network in the given environment.  One point is awarded
for each time instance in which the pole remains balanced.

@param: nn Nueral Network to be tested
@param: env testing environment
@return: fitness score
"""
def evaluateFitness(nn, env):
    observation = env.reset()
    for t in range(MAX_TIMESTEPS):
        #env.render()
        action = nn.makeDecision(observation)
        observation, reward, done, info = env.step(action)
        if done:
            return t + 1
            break
    return MAX_TIMESTEPS


"""
Calculates the fitness for each network in the population.
Builds a selection pool by adding (Fitness - 9) objects of each network
to the selectionpool array where fitness is the fitness score.
Calculates the average fitness of the generation

@param population some generation of nueral networks
@return two element array.  first element is the selectionPool array.
Second element is the average fitness score.
"""
def evaluateGeneration(population):
    selectionPool = []

    #avg = 0

    best = [population[0], 0]
    
    threshhold = 9
    env = gym.make('CartPole-v1')
    
    for i in range(len(population)):
        
        fitness = evaluateFitness(population[i], env)

        if (fitness > best[1]):
            best[0] = population[i]
            best[1] = fitness
        
        #f.write(f"{fitness}, ")
        #avg += fitness
        #print("Network {} earned fitness score of {}".format(i, fitness))
        if (fitness > threshhold):
            for j in range(fitness - threshhold):
                selectionPool.append(population[i])
                
    env.close()

    #avg /= len(population)

    return [selectionPool, best]

"""
Creates a new generation by randomly selecting two parents fromthe selectionPool
and mating them together

@param selectionPool array ontaining more elements of the more fit networks and less
elements of less fit networks
@return next generation of nueral networks
"""
def buildNextGen(selectionPool):
    population = []
    for i in range(POPSIZE):
        x = random.randint(0, len(selectionPool) - 1)
        y = random.randint(0, len(selectionPool) - 1)
        population.append(mate(selectionPool[x], selectionPool[y]))
    return population


"""
Initialize a population, evaluate and generate the next generation for a set
number of generations
"""
def train():
    
    pop = initializePopulation()
    env = gym.make('CartPole-v1')
    
    for i in range(MAX_GEN):
        evaluationResults = evaluateGeneration(pop)
        pool = evaluationResults[0]
        demoNet = evaluationResults[1]

        observation = env.reset()
        print(f"Rendering best of Gen {i}")
        x = 0
        for t in range(MAX_TIMESTEPS):
            env.render()
            action = pool[-1].makeDecision(observation)
            observation, reward, done, info = env.step(action)
            if done:
                x = 1
                print("Score: {}".format(t))
                print()
                break
        if (x == 0):
            print("Score: 200")
            print()
            break
            
        
        pop = buildNextGen(pool)
    env.close()


"""
Crossover selection technique.  For each weight and bias of the child network,
one of the parents is randomly selected to pass on the trait.  There is an additional
(MR / 100) % applied mutation rate, so that every MR weight or bias is assigned a
randomvalue.

@param nn1: The first parent network
@param nn2: second parent network
@return: child of the two networks
"""
def mate(nn1, nn2):
    wih = np.empty(nn1.wih.shape)
    for i in range(wih.shape[0]):
        for j in range(wih.shape[1]):
            if (random.randint(0, MR) == 0): 
                wih[i][j] = random.random()
            elif (random.randint(0, 1) == 0):
                wih[i][j] = nn1.wih[i][j]
            else:
                wih[i][j] = nn2.wih[i][j]
            
    who = np.empty(nn1.who.shape)
    for i in range(who.shape[0]):
        for j in range(who.shape[1]):
            if (random.randint(0, MR) == 0):
                who[i][j] = random.random()
            elif (random.randint(0, 1) == 0):
                who[i][j] = nn1.who[i][j]
            else:
                who[i][j] = nn2.who[i][j]
            
                
    b0 = np.empty((nn1.b0.size,))
    for i in range(b0.size):
        if (random.randint(0, MR) == 0):
            b0[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b0[i] = nn1.b0[i]
        else:
            b0[i] = nn2.b0[i]
        

    b1 = np.empty((nn1.b1.size,))
    for i in range(b1.size):
        if (random.randint(0, MR) == 0):
            b1[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b1[i] = nn1.b1[i]
        else:
            b1[i] = nn2.b1[i]
        

    b2 = np.empty((nn1.b2.size,))
    for i in range(b2.size):
        if (random.randint(0, MR) == 0):
            b2[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b2[i] = nn1.b2[i]
        else:
            b2[i] = nn2.b2[i]
        
    return NueralNetwork(wih, who, b0, b1, b2)



train()
        
