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

        self.weights = np.random.rand(3)
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
