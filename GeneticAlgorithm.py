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
    for t in range(200):
        #env.render()
        action = nn.makeDecision(observation)
        observation, reward, done, info = env.step(action)
        if done:
            return t + 1
            break
    return 200

def buildSelectionPool(population):
    selectionPool = []

    best = [population[0], 0]
    
    threshhold = 9
    env = gym.make('CartPole-v1')
    for i in range(len(population)):
        fitness = evaluateFitness(population[i], env)
        if (fitness > best[1]):
            best[0] = population[i]
            best[1] = fitness
        #print("Network {} earned fitness score of {}".format(i, fitness))
        if (fitness > threshhold):
            for j in range(fitness - threshhold):
                selectionPool.append(population[i])
    selectionPool.append(best[0])
                                     
    env.close()

    return selectionPool

def buildNextGen(selectionPool):
    population = []
    for i in range(POPSIZE):
        x = random.randint(0, len(selectionPool) - 1)
        y = random.randint(0, len(selectionPool) - 1)
        population.append(mate(selectionPool[x], selectionPool[y]))
    return population

def mate(nn1, nn2):
    wih = np.empty(nn1.wih.shape)
    for i in range(wih.shape[0]):
        for j in range(wih.shape[1]):
            if (random.randint(0, 100) == 0): 
                wih[i][j] = random.random() #random mutation
            elif (random.randint(0, 1) == 0):
                wih[i][j] = nn1.wih[i][j]
            else:
                wih[i][j] = nn2.wih[i][j]
            
    
    who = np.empty(nn1.who.shape)
    for i in range(who.shape[0]):
        for j in range(who.shape[1]):
            if (random.randint(0, 100) == 0):
                who[i][j] = random.random()
            elif (random.randint(0, 1) == 0):
                who[i][j] = nn1.who[i][j]
            else:
                who[i][j] = nn2.who[i][j]
            
                
    b0 = np.empty((nn1.b0.size,))
    for i in range(b0.size):
        if (random.randint(0, 100) == 0):
            b0[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b0[i] = nn1.b0[i]
        else:
            b0[i] = nn2.b0[i]
        

    b1 = np.empty((nn1.b1.size,))
    for i in range(b1.size):
        if (random.randint(0, 100) == 0):
            b1[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b1[i] = nn1.b1[i]
        else:
            b1[i] = nn2.b1[i]
        

    b2 = np.empty((nn1.b2.size,))
    for i in range(b2.size):
        if (random.randint(0, 100) == 0):
            b2[i] = random.random()
        elif (random.randint(0, 1) == 0):
            b2[i] = nn1.b2[i]
        else:
            b2[i] = nn2.b2[i]
        
    

    return NueralNetwork(wih, who, b0, b1, b2)

def train():
    pop = initializePopulation()
    threshhold = 9

    reached_goal = 0

    i = 0
    while(reached_goal == 0):
        print()
        print("Rendering most fit of Gen {}".format(i))
        pool = buildSelectionPool(pop)
        pop = buildNextGen(pool)

        env = gym.make('CartPole-v1')
        observation = env.reset()
        for t in range(200):
            env.render()
            action = pool[-1].makeDecision(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Score: {}".format(t))
                print()
                break
            if (t == 199):
                reached_goal = 1
                print()
                print("Goal state reached after {} generations".format(i))
        env.close()
        i += 1
        

train()
        
