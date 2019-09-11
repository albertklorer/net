import numpy as np

# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

# inherit from base class
class DenseLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        print(self.weights)
        self.bias = np.random.rand(1, output_size) - 0.5
        print(self.bias)

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        # 
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# inherit from base class
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # returns the prime activated input * error
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# tanh activation function
def tanh(x):
    return np.tanh(x)

# derivative of tanh
def tanh_prime(x):
    return 1 - np.tanh(x)**2

# mean squared errror loss function
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

# derivative of mean squared error
def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss 
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

network = Network()
network.add(DenseLayer(3,3)) 
network.add(ActivationLayer(tanh, tanh_prime)) 
network.add(DenseLayer(3,4)) 
network.add(ActivationLayer(tanh, tanh_prime))
network.add(DenseLayer(4,2))

network.use(mse, mse_prime)

x_train = np.array([[[0,0,0]], [[0,0,1]], [[0,1,0]], [[0,1,1]], [[1,0,0]], [[1,0,1]], [[1,1,0]], [[1,1,1]]])
y_train = np.array([[[1,0]], [[0,1]], [[0,1]], [[1,0]], [[0,1]], [[1,0]], [[1,0]], [[1,0]]])

network.fit(x_train, y_train, epochs=1, learning_rate=5.0)

print(network.layers[2].weights)




