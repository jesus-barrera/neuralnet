 # -*- coding: utf-8 -*-

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import numpy as np

class Plotter():
    def __init__(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.sampling = False

        self.init_axes()

        self.function_plt, = self.space_axes.plot([], [], 'b-', label='f(x)')
        self.approximated_plt, = self.space_axes.plot([], [], 'g--', label='Aprox.')
        self.samples_plt, = self.space_axes.plot([], [], 'ro', label='Muestras')

        self.space_axes.legend()

        # canvas events
        self.canvas.mpl_connect('button_press_event', self.on_button_press)

    def init_axes(self):
        # input space axes
        axes = self.figure.add_subplot(211)

        axes.set_xlabel('x')
        axes.set_ylabel('y')

        axes.set_xlim(-5, 5)
        axes.set_ylim(-5, 5)

        self.space_axes = axes

        # gauss axes
        axes = self.figure.add_subplot(212)

        axes.set_xlim(-5, 5)
        axes.set_ylim(-0.5, 1.5)

        self.gauss_axes = axes

    def on_button_press(self, event):
        if event.xdata and self.sampling:
            x, y = self.samples_plt.get_data()

            x = np.append(x, event.xdata)
            y = np.append(y, event.ydata)

            self.samples_plt.set_data(x, y)

            self.canvas.draw()

    def plot_function(self, x, y):
        self.function_plt.set_data(x, y)
        self.canvas.draw()

    def plot_approximated(self, x, y):
        self.approximated_plt.set_data(x, y)
        self.canvas.draw()

    def plot_samples(self, x, y):
        self.samples_plt.set_data(x, y)
        self.canvas.draw()
