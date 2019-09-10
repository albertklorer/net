import numpy as np 
from tqdm import trange

np.random.seed(42)

class Layer:
    def __init__(self):
        pass
    
    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        
        # identity matrix
        d_layer_d_input = np.eye(num_units)
        
        # chain rule
        return np.dot(grad_output, d_layer_d_input) 

class ReLU(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        pass
    
    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class TanH(Layer):
    # applies elementwise tanh to all inputs
    def __init__(self):
        pass

    def forward(self, input):
        # apply elementwise to all inputs
        tanh_forward = np.tanh(input)
        return tanh_forward

    def backward(self, imput, grad_output):
        tanh_grad = 1.0 - np.tanh(input)**2
        return grad_output * tanh_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        
        return np.dot(input,self.weights) + self.biases
    
    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

def softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / logits.shape[0]

def train(network,X,y):
    # Train our network on a given batch of X and y.
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)

    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        
    return np.mean(loss)

def forward(network, X):
    # Compute activations of all network layers by applying them sequentially.
    # Return a list of activations for each layer. 
    
    activations = []
    input = X
    # Looping through each layer
    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    # Compute network predictions. Returning indices of largest Logit probability
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# loading variables with training data for - numerical validation
X_train = np.array([[0.0,0.0,0.0], [0.0,0.0,1.0], [0.0,1.0,0.0]])
y_train = np.array([[1.0, 0.0], [0.0, 1.0], [[0.0, 1.0]]])
X_val = np.array([[0.0, 1.0, 1.0]])
y_val = np.array([[1.0, 0,0]])
X_test = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
y_test = np.array([[0.0, 1.0], [1.0, 0.0]])

# loading variables with initialization weights and biases
h1_weights = np.array([[0.1, 0.2, 0.3], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]).T
h1_biases = np.array([0.2, 0.1, 0.9])

h2_weights = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]).T
h2_biases = np.array([0.0, 0.2, 0.0, -0.1])

o_weights = np.array([[1.5, 1.2, 1.0, 0.0], [0.0, 0.8, 0.1, 0.0]]).T
o_biases = np.array([-0.2, -0.1])

# create network layer by layer
network = []
network.append(Dense(3,3, learning_rate=5.0))
network.append(TanH())
network.append(Dense(3,4, learning_rate=5.0))
network.append(TanH())
network.append(Dense(4,2, learning_rate=5.0))

# initialize networks with weights and biases

print(network[0].weights)
print(network[0].biases)

print(network[2].weights)
print(network[2].biases)

print(network[4].weights)
print(network[4].biases)

network[0].weights = h1_weights
network[0].biases = h1_biases

network[2].weights = h2_weights
network[2].biases = h2_biases

network[4].weights = o_weights
network[4].biases = o_biases

print("INITIALIZED ")
print(network[0].weights)
print(network[0].biases)

print(network[2].weights)
print(network[2].biases)

print(network[4].weights)
print(network[4].biases)

# update network with biases and weights

train_log = []
val_log = []
for epoch in range(1):
    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=100,shuffle=True):
        train(network,x_batch,y_batch)
    
    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))

    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])

