import numpy as np
import math
import random

class RbfNeuron():
    def __init__(self, mean):
        self.patterns = []
        self.mean = mean
        self.width = None

    def gaussian(self, pattern):
        return math.exp(- sum( (pattern - self.mean)**2 )
                        / (2 * self.width**2))

class RbfNetwork():
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self._num_inputs = num_inputs
        self._num_hidden = num_hidden
        self._num_outputs = num_outputs

        self._weights = None
        self._rbf_neurons = None
        self._rbf_outputs = None
        self._net_outputs = None

        self._inputs = None
        self._outputs = None
        self._p_nearest = 2

    def test(self, pattern):
        self._feedforward(pattern)

        return self._net_outputs

    def train(
            self,
            inputs,
            outputs,
            learning_rate,
            max_epochs,
            goal_error,
            update):

        self._inputs = inputs
        self._outputs = outputs

        self._kmeans_clustering()

        for neuron in self._rbf_neurons:
            self._set_neuron_width(neuron)

        self._find_weights(learning_rate, max_epochs, goal_error, update)

    def _feedforward(self, pattern):
        outputs = [neuron.gaussian(pattern) for neuron in self._rbf_neurons]
        outputs.insert(0, 1) # insert constant threshold input

        self._rbf_outputs = np.array(outputs)
        self._net_outputs = self._weights.dot(self._rbf_outputs)

    def _kmeans_clustering(self):
        '''K-Means Clustering algorithm.'''

        # initialize RBF neurons with random centers
        self._rbf_neurons = [RbfNeuron(np.random.uniform(-5, 5, self._num_inputs))
                             for i in range(self._num_hidden)]

        assigments = []

        # assign each pattern to a neuron
        for pattern in self._inputs:
            nearest = self._find_nearest_neuron(pattern, self._rbf_neurons)
            assigments.append((pattern, nearest))

        self._update_means(assigments)

        done = False

        while not done:
            reassigned = 0
            new_assigments = []

            # reassign patterns
            for pattern, current in assigments:
                nearest = self._find_nearest_neuron(pattern, self._rbf_neurons)

                if nearest != current:
                    reassigned += 1

                new_assigments.append((pattern, nearest))

            assigments = new_assigments

            if reassigned > 0:
                self._update_means(assigments)
            else:
                done = True

    def _find_nearest_neuron(self, pattern, neurons):
        '''Finds the nearest neuron.'''

        distances = [(n, np.linalg.norm(pattern - neuron.mean))
                     for n, neuron in enumerate(neurons)]

        nearest, distance = min(distances, key=lambda x: x[1])

        return nearest

    def _update_means(self, assigments):
        # empty neurons
        for neuron in self._rbf_neurons:
            neuron.patterns = []

        # insert each pattern into its assigned neuron
        for pattern, n in assigments:
            self._rbf_neurons[n].patterns.append(pattern)

        # update means
        for neuron in self._rbf_neurons:
            if len(neuron.patterns) > 0:
                neuron.mean = sum(neuron.patterns) / len(neuron.patterns)

    def _set_neuron_width(self, neuron):
        '''Sets a RBF neuron width.

        The width is determined by the mean distance between the neuron and its
        p nearest centers.
        '''
        neurons = self._rbf_neurons[:]
        neurons.remove(neuron)

        total = 0
        count = 0

        while count < self._p_nearest and len(neurons) > 0:
            n = self._find_nearest_neuron(neuron.mean, neurons)
            nearest = neurons.pop(n)

            total += np.linalg.norm(neuron.mean - nearest.mean)
            count += 1

        neuron.width = total / count

    def _find_weights(self, learning_rate, max_epochs, goal_error, update):
        rows = self._num_outputs
        cols = self._num_hidden + 1 # threshold

        # initialize weights in [-1, 1]
        self._weights = np.random.uniform(-1, 1, (rows, cols))
        update()

        done = False
        epochs = 0

        while epochs < max_epochs and not done:
            total_error = 0

            for i in range( len(self._inputs) ):
                pattern = self._inputs[i]
                desired = self._outputs[i]

                output = self.test(pattern)

                # compute error
                error = desired - output
                total_error += error.dot(error) / 2

                # update weights
                error = error.reshape( (len(error), 1) )
                rbf_outputs = self._rbf_outputs.reshape( (1, len(self._rbf_outputs)) )

                self._weights += learning_rate * error.dot(rbf_outputs)

            epochs += 1
            mean_error = total_error / len(self._inputs)

            if mean_error <= goal_error:
                done = True

            update()

        return done, epochs


if __name__ == '__main__':
    rbfn = RbfNetwork(2, 4, 1)

    # The XOR problem
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])

    outputs = np.array([[0], [1], [0], [1]])

    rbfn.train(inputs, outputs)

    for test in inputs:
        out = rbfn.test(test)
        print '{} -> {}'.format(test, out)
