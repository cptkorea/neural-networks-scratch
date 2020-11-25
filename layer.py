from nn_math import *

class Layer:
    activation_map = {'Sigmoid' : (Sigmoid, dSigmoid),
                      'ReLU' : (ReLU, dReLU)}
    scale_map = {'Sigmoid': lambda s: np.sqrt(2/s),
                 'ReLU' : lambda s: np.sqrt(2/s)} # He Initialization

    def __init__(self, size, activation=None, use_bias=True):
        self.size = size
        self.activation = activation
        self.num_inputs = -1
        self.use_bias = use_bias

    def generate_weights(self):
        r = 1.0 # Default scaling
        if self.activation in Layer.scale_map:
            r = Layer.scale_map[self.activation](self.size)
        self.weights = np.random.randn(self.num_inputs, self.size) * r
        if self.use_bias:
            self.bias = np.zeros((1, self.size))
        else:
            self.bias = np.zeros(1, self.size)

    def activation_fn(self):
        if self.activation in Layer.activation_map:
            return lambda x: Layer.activation_map[self.activation][0](x)
        else:
            return lambda x: x

    def dActivation_fn(self):
        if self.activation in Layer.activation_map:
            return lambda x: Layer.activation_map[self.activation][1](x)
        else:
            return lambda x: np.ones(x.shape)

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

