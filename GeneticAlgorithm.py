import numpy as np
import random
from NueralNetwork import NueralNetwork
import gym


POPSIZE = 16

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
    for t in range(250):
        env.render()
        action = nn.makeDecision(observation)
        observation, reward, done, info = env.step(action)
        if done:
            return t + 1
            break
    return 250

def buildSelectionPool(population):
    selectionPool = [[0,0], [0,0], [0,0], [0,0]]
    env = gym.make('CartPole-v1')
    for i in range(len(population)):
        fitness = evaluateFitness(population[i], env)
        print("Network {} earned fitness score of {}".format(i, fitness))
        if (fitness > selectionPool[3][1]):
            selectionPool[2][1] = selectionPool[3][1]
            selectionPool[1][1] = selectionPool[2][1]
            selectionPool[0][1] = selectionPool[1][1]
            selectionPool[3] = [i, fitness]
        if (fitness > selectionPool[2][1]):
            selectionPool[1][1] = selectionPool[2][1]
            selectionPool[0][1] = selectionPool[1][1]
            selectionPool[2] = [i, fitness]
        if (fitness > selectionPool[1][1]):
            selectionPool[0][1] = selectionPool[1][1]
            selectionPool[1] = [i, fitness]
        if (fitness > selectionPool[0][1]):
            selectionPool[0] = [i, fitness]
    env.close()

    poolList = []
    for i in range(len(selectionPool)):
        poolList.append(population[selectionPool[i][0]])
    return poolList
        

    #Build selection pool from networks scoring above average
    threshhold = 0
    for i in range(len(population)):
        threshhold += populationFitness[i]
    threshhold /= len(populationFitness)
    print(threshhold)

    selectionPool = []
    for i in range(len(population)):
        if(populationFitness[i] > threshhold):
            for k in range(int((populationFitness[i] - threshhold))):
                selectionPool.append(population[i])
                print(populationFitness[i])
                      
    return selectionPool

def buildNextGen(selectionPool):
    population = []
    population.append((selectionPool[3]))
    for i in range(POPSIZE - 1):
        x = random.randint(0, len(selectionPool) - 1)
        y = random.randint(0, len(selectionPool) - 1)
        population.append(mate(selectionPool[x], selectionPool[y]))
        """
    for i in range(int(POPSIZE * 1/8)):
        wih = np.random.random((4,24))
        who = np.random.random((24,2))
        b0 = np.random.random((4,))
        b1 = np.random.random((24,))
        b2 = np.random.random((2,))
        population.append(NueralNetwork(wih,who,b0,b1,b2))
        """
    return population

def mate(nn1, nn2):
    wih = np.empty(nn1.wih.shape)
    for i in range(wih.shape[0]):
        for j in range(wih.shape[1]):
            if (random.randint(0, 48) == 0): 
                wih[i][j] = random.random() #random mutation
            elif (random.randint(0, 1) == 0):
                wih[i][j] = nn1.wih[i][j]
            else:
                wih[i][j] = nn2.wih[i][j]
            
    
    who = np.empty(nn1.who.shape)
    for i in range(who.shape[0]):
        for j in range(who.shape[1]):
            if (random.randint(0, 24) == 0):
                who[i][j] = random.random()
            elif (random.randint(0, 1) == 0):
                who[i][j] = nn1.who[i][j]
            else:
                who[i][j] = nn2.who[i][j]
            
                
    b0 = np.empty((nn1.b0.size,))
    for i in range(b0.size):
        if (random.randint(0, 8) == 0):
            b0[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b0[i] = nn1.b0[i]
        else:
            b0[i] = nn2.b0[i]
        

    b1 = np.empty((nn1.b1.size,))
    for i in range(b1.size):
        if (random.randint(0, 48) == 0):
            b1[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b1[i] = nn1.b1[i]
        else:
            b1[i] = nn2.b1[i]
        

    b2 = np.empty((nn1.b2.size,))
    for i in range(b2.size):
        if (random.randint(0, 8) == 0):
            b2[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b2[i] = nn1.b2[i]
        else:
            b2[i] = nn2.b2[i]
        
    

    return NueralNetwork(wih, who, b0, b1, b2)

def train():
    pop = initializePopulation()
    threshhold = 9
    for i in range(100):
        print()
        print()
        print("Evaluating Gen {}".format(i))
        print()
        print()
        pool = buildSelectionPool(pop)
        pop = buildNextGen(pool)

train()
        
