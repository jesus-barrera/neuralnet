 # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from neuralnet.adaline import Adaline

class Classifier:
    def __init__(self):
        # perceptron parameters
        self.learning_rate = 1
        self.max_epochs = 100
        self.min_error = 0.01
        self.training_set = []
        self.perceptron = Adaline(2) # two dimensional adaline

        self.figure = plt.figure()

        # create axes
        self.space_axes = self.init_space_axes()
        self.error_axes = self.init_error_axes()

        # add plots
        self.error_line, = self.error_axes.plot([0], [0], 'g-')
        self.separating_line, = self.space_axes.plot([0], [0], 'b-')

        # canvas properties
        self.figure.canvas.set_window_title(u'Perceptrón')
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)

        self.set_gui()

    def init_space_axes(self):
        axes = self.figure.add_axes([0.05, 0.2, 0.4, 0.75])

        axes.set_xlabel('x')
        axes.set_ylabel('y')

        axes.set_xlim(-10, 10)
        axes.set_ylim(-10, 10)

        return axes


    def init_error_axes(self):
        axes = self.figure.add_axes([0.5, 0.65, 0.3, 0.30])

        axes.set_xlabel(u'Época')
        axes.set_ylabel('Error')

        # x lim is set dynamically according to max_epochs
        axes.set_ylim(0, 1)

        return axes


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
                plt.axes([0.12, 0.11, 0.3, 0.03]),
                u'Razón aprendizaje',
                0.1, 2,
                valinit=self.learning_rate)

        self.learning_slider.on_changed(self.update_learning_rate)

        # max epochs slider
        self.epochs_slider = Slider(
                plt.axes([0.12, 0.07, 0.3, 0.03]),
                u'Máx. épocas',
                10, 500,
                valinit=self.max_epochs,
                valfmt='%d')

        self.epochs_slider.on_changed(self.update_max_epochs)

        # min error slider
        self.error_slider = Slider(
                plt.axes([0.12, 0.03, 0.3, 0.03]),
                u'Error Mín.',
                0, 1,
                valinit=self.min_error)

        self.error_slider.on_changed(self.update_min_error)

        # test x slider
        self.x_slider = Slider(
                plt.axes([0.5, 0.07, 0.3, 0.03]),
                u'x',
                -10, 10,
                valinit=0)

        # y slider
        self.y_slider = Slider(
                plt.axes([0.5, 0.03, 0.3, 0.03]),
                u'y',
                -10, 10,
                valinit=0)

        # test button
        ax = self.figure.add_axes([0.85, 0.03, 0.1, 0.075])
        self.test_btn = Button(ax, 'Probar')
        self.test_btn.on_clicked(self.test)

        # important! set current axes
        plt.sca(self.space_axes)

    def on_press(self, event):
        if event.xdata and event.inaxes == self.space_axes:
            if event.button is 1: # left button
                output = 1
            elif event.button is 3: # right button
                output = 0
            else:
                return

            x = event.xdata
            y = event.ydata

            # add new training pair; threshold is considered an input thus x0 = -1
            inputs = np.array([x, y])
            self.training_set.append((inputs, output))

            # plot new point
            self.plot_point(x, y, output)


    def update_learning_rate(self, rate):
        self.learning_rate = rate

    def update_max_epochs(self, num):
        self.max_epochs = int(num)

    def update_min_error(self, error):
        self.min_error = error

    def reset(self, event):
        self.training_set = []
        self.separating_line.set_data([0], [0])
        self.error_line.set_data([0], [0])
        self.space_axes.lines = [self.separating_line]

        self.figure.canvas.draw()

    def train_perceptron(self, event):
        self.before_training()

        converged, seasons = self.perceptron.train(
                self.training_set,
                self.learning_rate,
                self.max_epochs,
                self.min_error,
                self.update_separating_line,
                self.update_error_line)

        if not converged:
            print u'El perceptrón no convergió!'
        else:
            print u'El perceptrón convergió en {} épocas'.format(seasons)

    def test(self, event):
        x = self.x_slider.val
        y = self.y_slider.val

        inputs = np.array([x, y])
        output = self.perceptron.test(inputs)

        print 'Probar ({}, {}): {}'.format(x, y, output)

        self.plot_point(x, y, output)

    def plot_point(self, x, y, group):
        if group is 1:
            fmt = 'go'
        else:
            fmt = 'r^'

        line, = self.space_axes.plot([x], [y], fmt)
        line.figure.canvas.draw()


    def before_training(self):
        # reset error plot
        self.error_line.set_data([0], [0])
        self.error_axes.set_xlim(0, self.max_epochs)

        self.figure.canvas.draw()

    def update_separating_line(self, weights):
        w0, w1, w2 = weights

        x = np.array([-10, 10])
        y = (-w1 * x + w0) / w2

        self.separating_line.set_data(x, y)
        self.figure.canvas.draw()

    def update_error_line(self, error):
        # add new error
        y = self.error_line.get_ydata()
        y = np.append(y, error)

        x = np.arange(y.size)

        # update line
        self.error_line.set_data(x, y)
        self.figure.canvas.draw()

if __name__ == '__main__':
    c = Classifier()

    plt.show()
