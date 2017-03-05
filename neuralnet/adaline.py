import numpy as np

class Adaline:
    def __init__(self, input_size):
        self._input_size = input_size
        self._weights = None

        self._inputs = np.zeros(input_size + 1) # constant threshold input
        self._inputs[0] = -1

    def train(
            self,
            training_set,
            learning_rate,
            max_epochs,
            min_error,
            update_line,
            update_error):

        self._weights = 2 * np.random.rand(self._input_size + 1) - 1

        update_line(self._weights)

        epochs = 0

        while epochs < max_epochs:
            total_error = 0

            for inputs, desired in training_set:
                self._inputs[1:] = inputs

                output = self._sigmoid(np.dot(self._inputs, self._weights))
                derivative = output * (1 - output)
                error = desired - output

                self._weights += learning_rate * error  * self._inputs
                update_line(self._weights)

                total_error += error**2

            total_error /= len(training_set)
            epochs += 1

            update_error(total_error)

            if total_error <= min_error:
                converged = True
                break
        else:
            converged = False

        return converged, epochs

    def _sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def test(self, inputs):
        self._inputs[1:] = inputs

        if np.dot(self._inputs, self._weights) >= 0:
            return 1
        else:
            return 0
