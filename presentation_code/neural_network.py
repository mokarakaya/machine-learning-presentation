import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes # 3
        self.hidden_nodes = hidden_nodes # 3
        self.output_nodes = output_nodes # 1

        # input_nodes X hidden_nodes matrix
        self.weights_input_to_hidden = np.random.normal(size=(self.input_nodes, self.hidden_nodes))
        # hidden_nodes X output_nodes matrix
        self.weights_hidden_to_output = np.random.normal(size=(self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))  # sigmoid.

    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_outputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):

        output_error_term = y - final_outputs

        hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
        # derivative of sigmoid function is f(x) * (1 - f(x))
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        delta_weights_i_h += hidden_error_term * X[:, None]
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
input_nodes = 3
hidden_nodes = 3
output_nodes = 1
iterations = 10000
learning_rate = 0.3

X =np.array([[2, 3, 3], [4, 3, 3], [4, 3, 7], [4, 3, 6]])
y = np.array([2, 7, 1, 3])

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
for i in range(iterations):
    nn.train(X, y)

print(nn.run(X))