 # -*- coding: utf-8 -*-

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import numpy as np

class Plotter():
    def __init__(self):
        self.training_set = []
        self.training_points = []
        self.test_points = []
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # create axes
        grid = gridspec.GridSpec(3, 1)
        grid.update(hspace=0.5)
        self.space_axes = self.init_space_axes(grid[0:2, 0])
        self.error_axes = self.init_error_axes(grid[2:3, 0])

        self.error_line, = self.error_axes.plot([0], [0], 'g-')
        self.separating_surface = None
        self.colorbar = None

        # canvas events
        self.canvas.mpl_connect('button_press_event', self.on_button_press)

    def init_space_axes(self, spec):
        axes = self.figure.add_subplot(spec)

        axes.set_xlabel('x')
        axes.set_ylabel('y')

        axes.set_xlim(-10, 10)
        axes.set_ylim(-10, 10)

        return axes

    def init_error_axes(self, spec):
        axes = self.figure.add_subplot(spec)

        axes.set_xlabel(u'Época')
        axes.set_ylabel('Error')

        axes.set_ylim(0, 0.5)
        axes.set_xlim(0, 100)

        return axes

    def on_button_press(self, event):
        if event.xdata and event.inaxes == self.space_axes:
            if event.button == 1: # left button
                output = 1
            elif event.button == 3: # right button
                output = 0
            else:
                return

            x = event.xdata
            y = event.ydata

            self.plot_point(x, y, output) # plot new point

            # add new training pair
            inputs = np.array([x, y])
            output = np.array([output])

            self.training_set.append((inputs, output))

    def plot_point(self, x, y, group, is_test=False):
        if group == 1:
            props = {'color': 'g', 'marker': 'o', 'markersize': 8}
        else:
            props = {'color': 'r', 'marker': '^', 'markersize': 8}

        if is_test:
            props['markeredgewidth'] = 0.5
            props['markeredgecolor'] = 'w'

        line, = self.space_axes.plot([x], [y], **props)

        self.canvas.draw()

        return line

    def plot_separating_surface(self, mlp, mesh_size=30):
        self.clear_separating_surface() # remove previous surface

        step = complex(0, mesh_size)

        # TODO: do not hardcode grid dimensions
        x, y = np.mgrid[-10:10:step, -10:10:step]

        space = np.linspace(-10, 10, mesh_size)

        colors = []

        for i in space:
            for j in space:
                pattern = np.append(i, j)

                result, = mlp.test(pattern, discretize=False)

                colors.append(result)

        colors = np.array(colors).reshape((mesh_size, mesh_size))

        self.separating_surface = self.space_axes.pcolormesh(
                x, y, colors,
                cmap='RdYlGn',
                vmin=0,
                vmax=1)

    def update_error_line(self, error):
        # add new error
        y = self.error_line.get_ydata()
        y = np.append(y, error)

        x = np.arange(y.size)

        # update line
        self.error_line.set_data(x, y)

    def clear(self):
        self.clear_error()
        self.clear_training_set()
        self.clear_separating_surface()

        self.canvas.draw()

    def clear_error(self):
        self.error_line.set_data([0], [0])

    def clear_training_set(self):
        self.training_set = []
        self.space_axes.lines = []

    def clear_separating_surface(self):
        if self.separating_surface:
            self.space_axes.collections.remove(self.separating_surface)
            self.separating_surface = None
