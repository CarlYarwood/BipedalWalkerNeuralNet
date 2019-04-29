import numpy as np
from scipy.special import expit


"""
A two layer nueral network
"""
class NueralNetwork:

    """
    Simple Constructor for three layer nueral network

    @param wih weights on links between input layer and hidden layer nodes
    @param who weights on lonks between hidden layer and output layer nodes
    @param b0 bias on input layer nodes
    @param b1 bias on hidden layer nodes
    @param b2 bias on output layer nodes
    """
    def __init__(self, wih, who, b0, b1, b2):
        self.wih = wih
        self.who = who
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        

    """
    Choose an output node based on some given observation

    @param observation nueral network input
    """
    def makeDecision(self, observation):
        pos = [0, 0]

        table = [np.zeros((len(self.b0),)), np.zeros((len(self.b1),)), np.zeros((len(self.b2),))]

        for i in range(self.b2.size):
            activation = self.node_calc_act(2, i, observation, table)
            if (activation > pos[1]):
                pos[1] = activation
                pos[0] = i
        return pos[0]


    """
    Calculate recursively the activation of any given node in the network

    @param layer 0, 1, or 2.  0 corresponds to input layer and 2 corresponds to output
    @param pos position of the node within the layer
    @param observation network input
    """
    def node_calc_act(self, layer, pos, observation, table):
        if(table[layer][pos] != 0):
            return table[layer][pos]
        
        act = 0
        
        if (layer == 2):
            for i in range(self.b1.size):
                act += self.node_calc_act(1, i, observation, table) * self.who[i][pos]
                
            act = expit(act + self.b2[pos])

            table[layer][pos] = act
            return act 
            
        elif (layer == 1):
            for i in range(self.b0.size):
                act += self.node_calc_act(0, i, observation, table) * self.wih[i][pos]

            act = expit(act + self.b1[pos])

            table[layer][pos] = act
            return act 

        else: #input layer
            act = expit(observation[pos] + self.b0[pos])
            table[layer][pos] = act
            return act 

