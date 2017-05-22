 # -*- coding: utf-8 -*-

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import numpy as np

class Plotter():
    def __init__(self, xlim, ylim):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.xlim = xlim
        self.ylim = ylim

        self.init_axes()

        self.samples_plt, = self.axes.plot([], [], 'r+')
        self.weights_patch = None

        # canvas events
        self.canvas.mpl_connect('button_press_event', self.on_button_press)

    def init_axes(self):
        axes = self.figure.add_subplot(111)

        axes.set_xlabel('x')
        axes.set_ylabel('y')

        axes.set_xlim(*self.xlim)
        axes.set_ylim(*self.ylim)

        self.axes = axes

    def add_samples(self, x, y):
        xdata, ydata = self.samples_plt.get_data()

        xdata = np.append(xdata, x)
        ydata = np.append(ydata, y)

        self.plot_samples(xdata, ydata)

    def on_button_press(self, event):
        if event.xdata:
            self.add_samples(event.xdata, event.ydata)

    def plot_weights(self, som):
        self.remove_patch()

        vertices = []
        codes = []

        for row in range(som._rows):
            for col in range(som._cols):
                from_vertex = som.get_weights_at(row, col)

                if col < som._cols - 1:
                    self.draw_line(from_vertex, som.get_weights_at(row, col + 1), vertices, codes)

                if row < som._rows - 1:
                    self.draw_line(from_vertex, som.get_weights_at(row + 1, col), vertices, codes)

        self.weights_patch = PathPatch(Path(vertices, codes), linewidth=0.2)
        self.axes.add_patch(self.weights_patch)

        self.canvas.draw()

    def draw_line(self, from_vertex, to_vertex, vertices, codes):
        # move pen
        vertices.append(from_vertex)
        codes.append(Path.MOVETO)

        # draw line
        vertices.append(to_vertex)
        codes.append(Path.LINETO)

    def remove_patch(self):
        if self.weights_patch:
            self.weights_patch.remove()
            self.weights_patch = None

    def plot_samples(self, x, y):
        self.samples_plt.set_data(x, y)
        self.canvas.draw()

    def clear_canvas(self):
        self.samples_plt.set_data([], [])
        self.remove_patch()

        self.canvas.draw()
