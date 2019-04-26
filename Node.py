from scipy.special import expit

class Node:
    def __init__(self, prevNode, weight, bias):
        self.prevNode = prevNode #list of nodes from previous layer
        self.weight = weight #list of weights for each link between this node and every node from previous layer
        self.bias = bias

    def setPrevNode(prevNode):
        self.prevNode = prevNode

    def makeDecision(self, observation):
        activation  = .0
        for i in range(len(self.prevNode)):
            activation = self.prevNode[i].makeDecision(observation) + self.weight[i]
        activation += self.bias
        activation = expit(activation) #Implementation of Sigmoid function
        return activation
