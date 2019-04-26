from scipy.special import expit

class FirstLayerNode:
    
    def __init__(self, position, bias):
        self.position = position
        self.bias = bias

    def makeDecision(self, observation):
        return expit(observation[self.position] + self.bias)

    
        
