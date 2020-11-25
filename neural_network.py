import numpy as np
from nn_math import *

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

    def train(self, inputs, outputs, epochs=2000, batch_size=64, loss='mse', lr=1e-6):
        if loss =='mse':
            loss_fn, dloss = mse, dmse
        n, d = inputs.shape
        for i in range(epochs):
            indices = np.random.choice(n, batch_size)
            self.back_prop(np.take(inputs, indices, axis=0), np.take(outputs, indices, axis=0), lr=lr)
            if i % (epochs // 5) == 0:
                print('loss = {}'.format(mse(outputs, self.forward(inputs, back_prop=False))))

    def back_prop(self, inputs, outputs, lr=1e-6):
        layer_outputs, layer_activations = self.forward(inputs, back_prop=True)
        num_layers, n = len(layer_outputs), len(layer_activations) - 1
        delta = [None] * (n+1)
        delta[n], W = dmse(outputs, layer_outputs[n-1]), np.eye(self.layers[-1].size)
        velocities = [None] * (n-1) # Nesterov Gradient Descent
        for l in range(n-1,0,-1):
            output, activation, layer = layer_outputs[l], layer_activations[l], self.layers[l+1]
            dSigma = np.apply_along_axis(layer.dActivation_fn(), 0, output)
            delta[l], W = np.multiply(delta[l+1].dot(W.T), dSigma), layer.weights
            velocities[i] = 0.9 * velocities[i] + lr * activation.T.dot(delta[l])
            layer.weights = layer.weights - lr * activation.T.dot(delta[l])
            layer.bias = layer.bias - lr * np.mean(delta[l], axis=0, keepdims=True)
        return velocities


