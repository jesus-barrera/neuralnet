# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from neuralnet import Perceptron

class Classifier:
    def __init__(self):
        # perceptron parameters
        self.learning_rate = 0.1
        self.max_seasons = 100
        self.training_set = []
        self.perceptron = Perceptron()

        # plotting
        self.figure = plt.figure()

        self.axes = self.figure.add_axes([0.1, 0.2, 0.7, 0.7])

        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')

        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-10, 10)

        self.separating_line, = self.axes.plot([0], [0], 'b-')

        self.figure.canvas.set_window_title(u'Perceptrón')
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)

        self.set_gui()

    def set_gui(self):
        # train button
        ax = self.figure.add_axes([0.85, 0.825, 0.1, 0.075])
        self.train_btn = Button(ax, 'Entrenar')
        self.train_btn.on_clicked(self.train_perceptron)

        # reset button
        ax = self.figure.add_axes([0.85, 0.7, 0.1, 0.075])
        self.reset_btn = Button(ax, 'Reiniciar')
        self.reset_btn.on_clicked(self.reset)

        # learning rate slider
        self.learning_slider = Slider(
                plt.axes([0.25, 0.08, 0.3, 0.03]),
                u'Razón aprendizaje',
                0.1, 0.9,
                valinit=self.learning_rate)

        self.learning_slider.on_changed(self.update_learning_rate)

        # max seasons slider
        self.seasons_slider = Slider(
                plt.axes([0.25, 0.03, 0.3, 0.03]),
                u'Máx. épocas',
                10, 500,
                valinit=self.max_seasons,
                valfmt='%d')

        self.seasons_slider.on_changed(self.update_max_seasons)

        # important! set current axes
        plt.sca(self.axes)

    def on_press(self, event):
        if event.xdata and event.inaxes == self.axes:
            if event.button is 1: # left button
                output = 1
                fmt = 'go'
            elif event.button is 3: # right button
                output = 0
                fmt = 'r^'
            else:
                return

            x = event.xdata
            y = event.ydata

            # add new training pair; threshold is considered an input thus x0 = -1
            inputs = np.array([-1, x, y])
            self.training_set.append((inputs, output))

            # plot new point
            line, = plt.plot([x], [y], fmt)
            line.figure.canvas.draw()


    def update_learning_rate(self, rate):
        self.learning_rate = rate

    def update_max_seasons(self, num):
        self.max_seasons = int(num)

    def reset(self, event):
        self.training_set = []
        self.separating_line.set_data([0], [0])
        self.axes.lines = [self.separating_line]

        self.figure.canvas.draw()

    def train_perceptron(self, event):
        converged, seasons = self.perceptron.train(
                self.training_set,
                self.learning_rate,
                self.max_seasons,
                self.update_line)

        if not converged:
            print u'El perceptrón no convergió!'
        else:
            print u'El perceptrón convergió en {} épocas'.format(seasons)

    def update_line(self, weights):
        w0, w1, w2 = weights

        x = np.array([-10, 10])
        y = (-w1 * x + w0) / w2

        self.separating_line.set_data(x, y)

        self.figure.canvas.draw()

if __name__ == '__main__':
    c = Classifier()

    plt.show()
