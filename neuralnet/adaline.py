import numpy as np

class Adaline:
    def __init__(self):
        self.weights = None

    def train(
            self,
            training_set,
            learning_rate,
            max_epochs,
            min_error,
            update_line,
            update_error):

        epochs = 0

        self.weights = np.random.rand(3)
        update_line(self.weights)

        while epochs < max_epochs:
            total_error = 0

            for inputs, desired in training_set:
                output = self.sigmoid(np.dot(inputs, self.weights))
                derivative = output * (1 - output)
                error = desired - output

                self.weights += learning_rate * error * derivative * inputs
                update_line(self.weights)

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

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def test(self, inputs):
        if np.dot(inputs, self.weights) >= 0:
            return 1
        else:
            return 0
