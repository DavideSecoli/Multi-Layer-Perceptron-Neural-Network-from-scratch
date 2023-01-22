import numpy as np 

class ActivationFunction:
    def __init__(self, activation, activation_deriv):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_deriv = activation_deriv

    def forward_prop(self, input):
        self.input = input
        return self.activation(self.input)

    def back_prop(self, gradient_output, learning_rate):
        return np.multiply(gradient_output, self.activation_deriv(self.input))
    

class Sigmoid(ActivationFunction):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_deriv(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_deriv)


class Softmax:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward_prop(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def back_prop(self, gradient_output, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, gradient_output)