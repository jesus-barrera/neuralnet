# -*- coding: utf-8 -*-

import numpy as np

class MultiLayerPerceptron:
    '''Multilayer perceptron implementation.'''

    def __init__(self, shape):
        '''Creates a multilayer perceptron.

        Parameters:
        -----------
        shape - an iterable describing the network architecture.
                e.g. (400,2,1) is a network with 400 inputs, a hidden layer with
                two neurons and a single neuron in the output layer.
        '''
        self._shape = shape
        self._outputs_array = []
        self._weights_array = []
        self._net_values = []

        # Initialize input layer
        self._weights_array.append(np.zeros(0)) # input layer has no weights
        self._net_values.append(np.zeros(0))    # nor net values (activation values)

        inputs = np.zeros(self._shape[0] + 1) # add constant input -1 for threshold
        inputs[0] = -1
        self._outputs_array.append(inputs)

        # Initialize hidden layers
        for layer in xrange(1, len(self._shape)):
            neurons = self._shape[layer]
            inputs = len(self._outputs_array[layer - 1])

            # Initialize weights with random values in [-1, 1]
            weights = 2 * np.random.rand(neurons * inputs) - 1
            weights = weights.reshape((neurons, inputs))

            self._weights_array.append(weights)

            # Initialize outputs; since this layer outputs serve as inputs for
            # the next layer, we add the required constant input of -1 for
            # threshold.
            outputs = np.zeros(neurons + 1)
            outputs[0] = -1
            self._outputs_array.append(outputs)

            # Initialize activation values
            nets = np.zeros(neurons)
            self._net_values.append(nets)

    def train(
            self,
            training_set,
            learning_rate,
            max_epochs,
            min_error,
            update):
        '''Surpervised training of the MLP using the backpropagation algorithm.

        Parameters
        ----------
        training_set - list of tuples (x, d) where x is the input for the network
                       and d is the desired response value. Both must be 1-D
                       numpy arrays. The desired response values must be either
                       0 or 1 (two classes)
        learning_rate - the learning rate.
        max_epochs - maximun number of epochs.
        min_error - error goal.

        Returns
        -------
        (converged, epochs) if the algorithm reach the 'min_error' goal it's said
                            to have converged, thus 'converged' is True. Otherwise,
                            if the algorithm ends when the 'max_epochs' limit is
                            reached, 'converged' is False. The total number of
                            epochs is returned in both cases.
        '''

        converged = False
        epochs = 0

        while epochs < max_epochs and not converged:
            total_error = 0

            for inputs, desired in training_set:
                # Forward pass
                self._feed_forward(inputs)

                # Compute error
                outputs = self._outputs_array[-1][1:] # take last, skip threshold
                error = desired - outputs
                total_error += error.dot(error) # squared error

                # Backward pass
                self._back_propagation(error, learning_rate)

            epochs += 1
            total_error /= len(training_set) # averange squared error

            update(epochs, total_error)

            if total_error <= min_error:
                converged = True

        return converged, epochs

    def _feed_forward(self, inputs):
        # Set network inputs; keep threshold value
        self._outputs_array[0][1:] = inputs

        for layer in xrange(1, len(self._shape)):
            # Previous layer outputs are current layer inputs
            inputs = self._outputs_array[layer - 1]

            # Compute activation values
            self._net_values[layer] = self._weights_array[layer].dot(inputs)

            # Compute outputs using activation function
            outputs = self._sigmoid(self._net_values[layer])
            self._outputs_array[layer][1:] = outputs # keep threshold input

    def _back_propagation(self, error, learning_rate):
        sensibilities = [0] * len(self._shape)

        # Compute output layer sensibility
        outputs = self._outputs_array[-1][1:]
        derivative = outputs * (1 - outputs)
        sensibility = derivative * error
        sensibility = sensibility.reshape((sensibility.size, 1))

        sensibilities[-1] = sensibility

        for layer in reversed(xrange(1, len(self._shape) - 1)):
            next_layer = layer + 1

            # Compute hidden layer sensibility
            outputs = self._outputs_array[layer][1:]
            derivative = outputs * (1 - outputs)
            derivative = np.diag(derivative)

            # from weights take all rows, skip first column (threshold inputs)
            weights = self._weights_array[next_layer][:,1:]

            sensibility = weights.T.dot(sensibilities[next_layer])
            sensibility = derivative.dot(sensibility)

            sensibilities[layer] = sensibility

        # Adjust weights
        for layer in xrange(1, len(self._shape)):
            inputs = self._outputs_array[layer - 1]
            inputs = inputs.reshape((1, inputs.size))

            sensibility = sensibilities[layer]

            self._weights_array[layer] += learning_rate * sensibility.dot(inputs)

    def test(self, inputs, discretize=True):
        '''Test the neural network with a given input.

        Parameters
        ----------
        inputs - 1-D numpy array.
        discretize - tells if the predicted output must be discretized to 0's
                     and 1's.

        Returns
        -------
        1-D numpy array with the predicted values
        '''

        self._feed_forward(inputs)

        if discretize:
            # is this correct?
            outputs = self._net_values[-1]
            outputs = map(self._step, outputs)

            return np.array(outputs)
        else:
            return self._outputs_array[-1][1:]

    def _sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def _step(self, value):
        if value >= 0:
            return 1
        else:
            return 0

if __name__ == '__main__':
    mlp = MultiLayerPerceptron((2, 3, 1))

    # The XOR problem
    training_set = [
        (np.array([0, 0]), np.array([0])),
        (np.array([0, 1]), np.array([1])),
        (np.array([1, 1]), np.array([0])),
        (np.array([1, 0]), np.array([1]))]

    converged, epochs = mlp.train(
            training_set,
            0.3,
            20000,
            0.005)

    if converged:
        print u'La red convergió en {} épocas'.format(epochs)

        print 'Test (0,0) =', mlp.test(np.array([0, 0]))
        print 'Test (1,0) =', mlp.test(np.array([1, 0]))
        print 'Test (1,1) =', mlp.test(np.array([1, 1]))
        print 'Test (0,1) =', mlp.test(np.array([0, 1]))
    else:
        print 'La red no convergió'
