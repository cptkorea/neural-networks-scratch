import numpy as np
import nn_math

class NeuralNetwork:
    def __init__(self, layers):
        for i in range(1, len(layers)):
            layers[i].num_inputs = layers[i-1].size
            layers[i].generate_weights()

        self.layers = layers
        self.output_dim = layers[-1].size

    def forward(self, inputs, back_prop=False):
        layer_input, outputs, activations = inputs, [], [inputs]
        for layer in self.layers[1:]:
            layer_input, activation = layer.forward(layer_input)
            outputs.append(layer_input)
            activations.append(activation)
        return (outputs, activations) if back_prop else layer_input

    def train(self, inputs, outputs, loss='mse'):
        if loss =='mse':
            loss_fn = nn_math.mse
        train_loss, tol = 10000, 0.001

        for i in range(1000):
            layer_outputs, layer_activations = self.forward(inputs, back_prop=True)
            # for output in layer_outputs:
            #     print(output.shape)
            num_layers, n = len(layer_outputs), len(layer_activations) - 1
            delta = [None] * (n+1)
            delta[n], dW = nn_math.dmse(outputs, layer_outputs[n-1]), np.eye(self.layers[-1].size)
            if i % 100 == 0:
                print('loss = {}'.format(nn_math.mse(outputs, layer_activations[n])))
            for l in range(n-1,0,-1):
                output, activation, layer = layer_outputs[l], layer_activations[l], self.layers[l+1]
                delta[l] = np.multiply(delta[l+1].dot(dW.T), nn_math.dReLU(output))
                dW = layer.weights
                layer.weights = layer.weights - 2e-4 * activation.T.dot(delta[l])
                layer.bias = layer.bias - 2e-4 * np.mean(delta[l], axis=0, keepdims=True)
class Layer:
    def __init__(self, size, activation=None, use_bias=True):
        self.size = size
        self.activation = activation
        self.num_inputs = -1
        self.use_bias = use_bias

    def generate_weights(self):
        self.weights = np.random.randn(self.num_inputs, self.size)
        if self.use_bias:
            self.bias = np.zeros((1, self.size))
        else:
            self.bias = np.zeros(1, self.size)

    def activation_fn(self):
        if self.activation == 'Sigmoid':
            return lambda x: nn_math.Sigmoid(x)
        elif self.activation == 'ReLU':
            return lambda x: nn_math.ReLU(x)
        else:
            return lambda x: x

    def dActivation_fn(self):
        if self.activation == 'Sigmoid':
            return lambda x: nn_math.dSigmoid(x)
        elif self.activation == 'ReLU':
            return lambda x: nn_math.dReLU(x)
        else:
            return lambda x: 1

    def forward(self, inputs):
        outputs = inputs.dot(self.weights) + self.bias
        activation = np.apply_along_axis(self.activation_fn(), 0, outputs)
        return [outputs, activation]

# Child Classes exit to make descriptive NN. Ex: NeuralNetwork([Input(4), Layer(5), Output(2)])
class Input(Layer):
    def __init__(self, size):
        super(Input, self).__init__(size)

class Output(Layer):
    def __init__(self, size):
        super(Output, self).__init__(size)


