import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = None

    def train(
            self,
            training_set,
            learning_rate,
            max_seasons,
            on_adjust):

        done = False
        seasons = 0

        self.weights = np.random.rand(3) # 2 dimensions and threshold
        on_adjust(self.weights)

        while not done and seasons < max_seasons:
            done = True

            for inputs, output in training_set:
                error = output - self.test(inputs)

                if error != 0:
                    done = False

                    self.weights += learning_rate * error * inputs
                    on_adjust(self.weights)

            seasons += 1

        return done, seasons

    def test(self, inputs):
        if np.dot(inputs, self.weights) >= 0:
            return 1
        else:
            return 0


class MultiLayerPerceptron:
    def __init__(self, input_size, shape):
        self.shape = shape
        self.output_vectors = []
        self.weights_array = []
        self.sensibilities = []

        weights = np.random.random((shape[0], input_size + 1))
        self.weights_array.append(weights)

        for i in xrange(1, len(shape)):
            weights = np.random.random((shape[i], shape[i - 1] + 1))
            self.weights_array.append(weights)

        for i, neurons in enumerate(shape):
            self.output_vectors.append(np.zeros(neurons + 1))

            self.sensibilities.append()

    def train(
            self,
            training_set,
            learning_rate=0.1
            min_error=0.1):

        done = False

        while not done:
            for inputs, output in training_set:
                self.feed_forward(inputs)

                last = len(self.shape) - 1
                error = output - self.output_vectors[last]

                self.back_propagation(error)



    def feed_forward(self, inputs):
        # first layer
        for neuron in range(self.shape[0]):
            activation_value = np.dot(inputs, self.weights_array[0][neuron])

            self.output_vectors[0][neuron] = self.sigmoid(activation_value)

        for layer in xrange(1, len(self.shape)):
            for neuron in range(self.shape[layer]):
                layer_inputs = self.output_vectors[layer - 1]
                activation_value = np.dot(layer_inputs, self.weights_array[layer][neuron])

                self.output_vectors[layer][neuron] = self.sigmoid(activation_value)

    def back_propagation(self, error):

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def diff_sigmoid(self, value):
        return self.sigmoid(value) * (1 - self.sigmoid(value))
