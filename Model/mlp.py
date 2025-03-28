from Model.model import Model
import numpy as np

class MLP(Model):
    def __init__(self):
        self.layer1_neurons = 4
        self.layer2_neurons = 2
        self.layer3_neurons = 1
        self.y_mean = 0
        self.y_std = 0

    def init_member(self, input_size, learning_rate):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.init_weights()

    def init_weights(self):
        # Xavier initialization for weights
        limit1 = np.sqrt(2. / (self.input_size + 1))
        limit2 = np.sqrt(2. / (self.layer1_neurons + 1))
        limit3 = np.sqrt(2. / (self.layer2_neurons + 1))
        
        self.weights_layer1 = np.random.uniform(-limit1, limit1, (self.layer1_neurons, self.input_size + 1))
        self.weights_layer2 = np.random.uniform(-limit2, limit2, (self.layer2_neurons, self.layer1_neurons + 1))
        self.weights_layer3 = np.random.uniform(-limit3, limit3, (self.layer3_neurons, self.layer2_neurons + 1))

    def sigmoid(self, vj):
        return 1 / (1 + np.exp(-vj))
    
    def relu(self, vj):
        return np.maximum(0, vj)
    
    def relu_deriative(self, vj):
        return np.where(vj >= 0, 1, 0)

    def leaky_relu(self, vj, alpha=0.01):
        return np.maximum(alpha * vj, vj)
    
    def leaky_relu_deriative(self, vj, alpha=0.01):
        return np.where(vj >= 0, 1, alpha)
    
    def mean_squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def forward(self, x):
        bias_x1 = np.insert(x, 0, -1)
        hidden_output1 = []
        for w in self.weights_layer1:
            hidden_output1.append(self.relu(np.dot(w.T, bias_x1)))

        bias_x2 = np.insert(hidden_output1, 0, -1)        
        hidden_output2 = []
        for w in self.weights_layer2:
            hidden_output2.append(self.relu(np.dot(w.T, bias_x2)))

        bias_x3 = np.insert(hidden_output2, 0, -1)
        output = np.dot(self.weights_layer3[0].T, bias_x3)
        
        return hidden_output1, hidden_output2, output

    def backward(self, x, hidden_output1, hidden_output2, output, y, lr):        
        # backpropagation of the output layer
        output_delta = []
        output_error = y - output
        output_delta.append(output_error)        
        # backpropagation of the hidden layer
        # layer 2
        layer2_delta = []
        for h in range(len(hidden_output2)):
            delta = hidden_output2[h] * self.relu_deriative(hidden_output2[h]) * np.sum(output_delta * self.weights_layer3[:, h+1])
            layer2_delta.append(delta)

        # layer 1
        layer1_delta = []
        for h in range(len(hidden_output1)):
            delta = hidden_output1[h] * self.relu_deriative(hidden_output1[h]) * np.sum(layer2_delta * self.weights_layer2[:, h+1])
            layer1_delta.append(delta)

        # update weights
        for h in range(len(output_delta)):
            self.weights_layer3[h] += lr * output_delta[h] * np.insert(hidden_output2, 0, -1)

        for h in range(len(hidden_output2)):
            self.weights_layer2[h] += lr * layer2_delta[h] * np.insert(hidden_output1, 0, -1)

        for h in range(len(hidden_output1)):
            self.weights_layer1[h] += lr * layer1_delta[h] * np.insert(x, 0, -1)


    def train(self, X, y, lr, epochs):
        # Normalize input data
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)
        loss_list = []
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                hidden_output1, hidden_output2, output = self.forward(X[i])
                self.backward(X[i], hidden_output1, hidden_output2, output, y[i], lr)

                # Calculate and print the loss
                # print(f"y: {y[i]}, output: {output}")
                total_loss += self.mean_squared_error(y[i], output)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(y):.4f}")
            loss_list.append(total_loss/len(y))
            # Check for NaN or instability
            # total_loss is Not a Number or too large
            if np.isnan(total_loss) or total_loss >= 1e6: 
                print("Training stopped due to NaN values.")
                break
        return loss_list

    def predict(self, inputs):
        # normalize input
        hidden_1, hidden_2, output = self.forward(inputs)
        output = np.clip((output * self.y_std) + self.y_mean, -40, 40)
        return output