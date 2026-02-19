import numpy as np

# 1. Define the Sigmoid Activation Function & its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # If x is already the output of sigmoid, the derivative is simple: x * (1 - x)
    return x * (1 - x)

# 2. The Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        # Shape: (2, 2) for hidden, (2, 1) for output
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
        
        # Initialize biases
        self.bias_hidden = np.random.uniform(size=(1, hidden_size))
        self.bias_output = np.random.uniform(size=(1, output_size))

    def forward(self, X):
        # Step 1: Input -> Hidden Layer
        # Z1 = X . W1 + b1
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # A1 = Sigmoid(Z1)
        self.hidden_output = sigmoid(self.hidden_input)

        # Step 2: Hidden -> Output Layer
        # Z2 = A1 . W2 + b2
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        # A2 = Sigmoid(Z2) -> This is our prediction
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y, output, learning_rate):
        # --- PHASE 1: Calculate Gradients (The "Blame" Game) ---
        
        # 1. Calculate Error at Output
        # Error = Target - Prediction
        output_error = y - output
        
        # 2. Calculate Output Delta (Gradient)
        # Apply Chain Rule: Error * Derivative of Activation
        output_delta = output_error * sigmoid_derivative(output)

        # 3. Calculate Error at Hidden Layer
        # How much did the hidden layer contribute to the output error?
        # We propagate the error BACKWARDS through the weights
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        
        # 4. Calculate Hidden Delta
        # Apply Chain Rule again for hidden layer activation
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # --- PHASE 2: Update Weights (Gradient Descent) ---
        
        # New Weight = Old Weight + (Input.T dot Delta) * Learning Rate
        
        # Update Hidden-to-Output Weights
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        # Update Input-to-Hidden Weights
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            # 1. Forward Pass
            output = self.forward(X)
            
            # 2. Backward Pass
            self.backward(X, y, output, learning_rate)
            
            # Print loss every 1000 epochs to verify learning
            if i % 1000 == 0:
                loss = np.mean(np.square(y - output)) # MSE
                print(f"Epoch {i}, Loss: {loss:.6f}")

# --- Testing the Network on XOR ---

if __name__ == "__main__":
    # XOR Input Data (4 examples, 2 features each)
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])

    # XOR Target Output (4 examples, 1 label each)
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # Create Network: 2 Input neurons, 2 Hidden neurons, 1 Output neuron
    nn = NeuralNetwork(2, 2, 1)

    print("Training...")
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    print("\nTesting after training:")
    print(f"Input: [0, 0] -> Prediction: {nn.forward(np.array([[0,0]]))[0][0]:.4f}")
    print(f"Input: [0, 1] -> Prediction: {nn.forward(np.array([[0,1]]))[0][0]:.4f}")
    print(f"Input: [1, 0] -> Prediction: {nn.forward(np.array([[1,0]]))[0][0]:.4f}")
    print(f"Input: [1, 1] -> Prediction: {nn.forward(np.array([[1,1]]))[0][0]:.4f}")


/*Output:

Training...
Epoch 0, Loss: 0.290180
Epoch 1000, Loss: 0.246786
Epoch 2000, Loss: 0.220290
Epoch 3000, Loss: 0.174588
Epoch 4000, Loss: 0.147414
Epoch 5000, Loss: 0.055508
Epoch 6000, Loss: 0.016799
Epoch 7000, Loss: 0.008729
Epoch 8000, Loss: 0.005704
Epoch 9000, Loss: 0.004177

Testing after training:
Input: [0, 0] -> Prediction: 0.0591
Input: [0, 1] -> Prediction: 0.9447
Input: [1, 0] -> Prediction: 0.9456
Input: [1, 1] -> Prediction: 0.0598*/
