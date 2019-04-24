import random

class NueralNetwork:


    def __init__(self):
        self.no_layers = random.randint(1, 5)

        self.no_nodes_per_layer = []
        self.no_nodes_per_layer[0] = 4
        for i in range(1, self.no_layers):
            # Assign each hidden layer a descending number of nodes
            # First hidden layer has a max of (2 ** no_layers + 4) nodes
            # Second hidden layer has a max of (2 ** no_layers + 2) nodes and so on...
            self.no_nodes_per_layer[self.no_layers - i] = random.randint(2 ** i + 2, 2 ** i + 4)
        self.no_nodes_per_layer[self.no_layers] = 1 #one output node

        #self.bias = [[0] * self.no_nodes_per_layer[i] for i in range(self.no_layers)]

        print(no_layers)
        for i in range (no_layers + 1):
            print(no_nodes_per_layer[i])

nn = NueralNetwork()
