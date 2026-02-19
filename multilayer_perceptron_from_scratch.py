import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size):
            self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
            self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

            self.bias_hidden = np.random.uniform(size=(1, hidden_size))
            self.bias_output = np.random.uniform(size=(1, output_size))

        def forward(self, x):
            self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            self.hidden_output = sigmoid(self.hidden_input)
            self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
            self.final_output = sigmoid(self.final_input)
            
            return self.final_output

        def backward(self, X, y, output, learning_ratae):
            output_error = y - output
            output_delta = output_error * sigmoid_derivative(output)
            hidden_error = output_delta.dot(self.hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

            self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

            self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
