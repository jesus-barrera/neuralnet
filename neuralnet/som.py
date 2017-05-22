import numpy as np

class SelfOrganizingMap:
    def __init__(self, input_size, rows, cols):
        self._input_size = input_size
        self._rows = rows
        self._cols = cols
        self._num_neurons = rows * cols

        # get neurons position in the lattice as numpy arrays, so we can calculate the distance
        # between neurons.
        self._positions = []

        for row in range(rows):
            for col in range(cols):
                self._positions.append( np.array([row, col]) )

    def train(
            self,
            inputs,
            epochs,
            neighborhood_radius,
            learning_rate,
            update):

        # initialize weights with random values
        self._weights = np.random.uniform(-1, 1, (self._num_neurons, self._input_size))

        initial_learning_rate = learning_rate
        initial_neighborhood_radius = neighborhood_radius

        t1 = epochs / np.log(initial_neighborhood_radius)
        t2 = float(epochs)

        for epoch in range(epochs):
            # update neighborhood radius
            neighborhood_radius = initial_neighborhood_radius * np.exp(- epoch / t1)

            # update learning rate
            learning_rate = initial_learning_rate * np.exp(- epoch / t2)

            print epoch, neighborhood_radius, learning_rate

            for pattern in inputs:
                best = self._find_best_match(pattern)

                # update weights
                for neuron, weights in enumerate(self._weights):
                    self._weights[neuron] += (learning_rate
                                              * self._neighborhood_distance(best, neuron, neighborhood_radius)
                                              * (pattern - weights))

            update()

    def get_weights_at(self, row, column):
        index = row * self._cols +  column

        return self._weights[index]


    def _find_best_match(self, pattern):
        best = 0
        min_distance = np.linalg.norm(pattern - self._weights[best])

        for neuron in range(1, self._num_neurons):
            distance = np.linalg.norm(pattern - self._weights[neuron])

            if distance < min_distance:
                best = neuron
                min_distance = distance

        return best

    def _neighborhood_distance(self, center, neuron, radius):
        diff = self._positions[neuron] - self._positions[center]

        return np.exp(-sum(diff**2) / (2 * radius**2))
