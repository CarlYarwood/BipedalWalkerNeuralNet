import random
import numpy as np
from Node import Node
from FirstLayerNode import FirstLayerNode

class NueralNetwork:
    SL0 = 4
    SL1 = 20
    SL2 = 2
    
    def __init__(self, firstLayer, hiddenLayer, outputLayer):
        
        self.firstLayer = firstLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        
        if (len(self.firstLayer) == 0):
            for i in range(self.SL0):
                self.firstLayer.append(FirstLayerNode(i, random.random()))

        if (len(self.hiddenLayer) == 0):
            for i in range(self.SL1):
                weight = np.random.random((self.SL0,)) #input-hiddenLayer link weights
                self.hiddenLayer.append(Node(self.firstLayer, weight, random.random()))

        if (len(self.outputLayer) == 0):
            for i in range(self.SL2):
                weight = np.random.random((self.SL1,))
                self.outputLayer.append(Node(self.hiddenLayer, weight, random.random()))

    def makeDecision(self, observation):
        if(self.outputLayer[1].makeDecision(observation) > self.outputLayer[0].makeDecision(observation)):
            return 1
        else:
            return 0

    def combineFirstLayerNodes(self, n1, n2):
        b = 0
        if (random.randint(0, 1) == 0):
            b = n1.bias
        else:
            b = n2.bias

        return Node(n1.position, b)

    def combineNodes(self, n1, n2):
        if (len(n1.weight) != len(n2.weight)): return

        
        if (random.randint(0, 1) == 0):
            b = n1.bias
        else:
            b = n2.bias

        w = []
        for i in range(len(N1.weight)):
            if (random.randint(0, 1) == 0):
                w.append(N1.weight[i])
            else:
                w.append(N2.weight[i])

        pn = []
        return Node(pn, w, b)



        
"""
fl = []
sl = []
tl = []
NN = NueralNetwork(fl,sl,tl)
print(NN.makeDecision([1, 2, 3, 4]))
"""
