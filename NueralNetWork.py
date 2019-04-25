import numpy as np
from scipy.special import expit


"""
A two layer nueral network
"""
class NueralNetwork:

    def __init__(self, wih, who, b0, b1, b2):
        self.wih = wih
        self.who = who
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

    def makeDecision(self, observation):
        pos = [0, 0]
        for i in range(self.b2.size):
            activation = self.node_calc_act(2, i, observation)
            if (activation > pos[1]):
                pos[1] = activation
                pos[0] = i
        return pos[0]

    def node_calc_act(self, layer, pos, observation):
        act = 0
        if (layer == 2):
            for i in range(self.b1.size):
                act += self.node_calc_act(1, i, observation) * who[i][pos]
                
            act += b2[pos]

            return expit(act) #scipy implementation of sigmoid funciton
            
        if (layer == 1):
            for i in range(self.b0.size):
                act += self.node_calc_act(0, i, observation) * wih[i][pos]

            return expit(act + b1[pos])

        else: #input layer
            return expit(observation[pos] + b0[pos])


wih = np.random.random((4,24))
who = np.random.random((24,2))
b0 = np.random.random((4,))
b1 = np.random.random((24,))
b2 = np.random.random((2,))

nn = NueralNetwork(wih,who,b0,b1,b2)
print(nn.makeDecision([0,1,2,3]))
