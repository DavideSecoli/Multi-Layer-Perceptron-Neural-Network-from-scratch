import numpy as np 

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward_prop(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def back_prop(self, gradient_output, learning_rate):
        gradient_weights = np.dot(gradient_output, self.input.T)
        gradient_input = np.dot(self.weights.T, gradient_output)
        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_output
        return gradient_input
