import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# classes

#dense layer
class Layer_Dense:

    #layer init
    def __init__(self, n_inputs, n_neurons):
        #random init of weights & biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    #fwrd pass
    def forward(self, inputs):
        # calc output vals from input, weights, biases
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:

    #frwd pass
    def forward(self,inputs):
        #calc output vals from input
        self.output = np.maximum(0,inputs)

class Activation_Softmax:

    #frwd pass
    def forward(self, inputs):
        #get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))

        #normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)

        self.output = probabilities

### end of classes

x,y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)

activation2 = Activation_Softmax()

dense1.forward(x)


activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output)

