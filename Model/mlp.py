from Model.model import Model
import numpy as np

class MLP(Model):
    def __init__(self, input_size, learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.init_weights()

    def init_weights(self):
        # hidden layer:
        # layer 1: 8 neurons; layer 2: 6 neurons
        # output:
        # layer 3: 1 neuron

        layer1_neurons = 8
        layer2_neurons = 6
        layer3_neurons = 1
        # Xavier initialization for weights
        self.weights_layer1 = np.random.randn(layer1_neurons, self.input_size + 1) * np.sqrt(2. / (self.input_size + 1))
        self.weights_layer2 = np.random.randn(layer2_neurons, layer1_neurons + 1) * np.sqrt(2. / (layer1_neurons + 1))
        self.weights_layer3 = np.random.randn(layer3_neurons, layer2_neurons + 1) * np.sqrt(2. / (layer2_neurons + 1))
        # self.weights_layer1 = np.array([np.random.uniform(-2, 2, self.input_size + 1) for i in range(layer1_neurons)])
        # self.weights_layer2 = np.array([np.random.uniform(-2, 2, layer1_neurons + 1) for i in range(layer2_neurons)])
        # self.weights_layer3 = np.array([np.random.uniform(-2, 2, layer2_neurons + 1) for i in range(layer3_neurons)])
        # print("Initial weights_layer1:", self.weights_layer1)
        # print("Initial weights_layer2:", self.weights_layer2)
        # print("Initial weights_layer3:", self.weights_layer3)

    def sigmoid(self, vj):
        return 1 / (1 + np.exp(-vj))

    def leaky_relu(self, vj, alpha=0.01):
        return np.maximum(alpha * vj, vj)
    
    def mean_squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def forward(self, x):
        bias_x1 = np.insert(x, 0, -1)
        hidden_output1 = []
        for w in self.weights_layer1:
            # print(w)
            # print(self.sigmoid(np.dot(w.T, bias_x1)))
            hidden_output1.append(self.leaky_relu(np.dot(w.T, bias_x1)))
        # print(hidden_output1)
        bias_x2 = np.insert(hidden_output1, 0, -1)        
        hidden_output2 = []
        for w in self.weights_layer2:
            hidden_output2.append(self.leaky_relu(np.dot(w.T, bias_x2)))
        # print(hidden_output2)
        bias_x3 = np.insert(hidden_output2, 0, -1)
        # print(bias_x3)
        # output = self.sigmoid(np.dot(self.weights_layer3[0].T, bias_x3))
        output = self.sigmoid(np.dot(self.weights_layer3[0].T, bias_x3))
        # print(output)
        return hidden_output1, hidden_output2, output

    def backward(self, x, hidden_output1, hidden_output2, output, y, lr):        
        # backpropagation of the output layer
        output_delta = []
        output_error = y - output
        output_delta.append(output_error * output * (1 - output))
        
        # backpropagation of the hidden layer
        # layer 2
        layer2_delta = []
        for h in range(len(hidden_output2)):
            delta = hidden_output2[h] * (1 - hidden_output2[h]) * np.sum(output_delta * self.weights_layer3[:, h+1])
            layer2_delta.append(delta)

        # layer 1
        layer1_delta = []
        for h in range(len(hidden_output1)):
            delta = hidden_output1[h] * (1 - hidden_output1[h]) * np.sum(output_delta * self.weights_layer2[:, h+1])
            layer1_delta.append(delta)

        # update weights
        for h in range(len(output_delta)):
            self.weights_layer3[h] += lr * output_delta[h] * np.insert(hidden_output2, 0, -1)

        for h in range(len(hidden_output2)):
            self.weights_layer2[h] += lr * layer2_delta[h] * np.insert(hidden_output1, 0, -1)

        for h in range(len(hidden_output1)):
            self.weights_layer1[h] += lr * layer1_delta[h] * np.insert(x, 0, -1)
        
        # print("Updated weights_layer3:", self.weights_layer2)
        # print("Updated weights_layer2:", self.weights_layer2)
        # print("Updated weights_layer1:", self.weights_layer1)

    def train(self, X, y, lr, epochs):
        # Normalize input data
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                hidden_output1, hidden_output2, output = self.forward(X[i])
                self.backward(X[i], hidden_output1, hidden_output2, output, y[i], lr)

                # Calculate and print the loss
                # print(f"y: {y[i]}, output: {output}")
                total_loss += self.mean_squared_error(y[i], output)
            
            
            print(f"Epoch {epoch}, Loss: {total_loss/len(y):.4f}")

            # Check for NaN or instability
            # total_loss is Not a Number or too large
            if np.isnan(total_loss) or total_loss >= 1e6: 
                print("Training stopped due to NaN values.")
                break

    def predict(self, front, left, right):
        input_data = np.array([[front, left, right]])
        return self.forward(input_data)[0, 0]

