 # -*- coding: utf-8 -*-
import numpy as np
from gi.repository import Gtk
from gi.repository.GdkPixbuf import Pixbuf
from neuralnet.rbfn import RbfNetwork
from plotting.rbfn import Plotter
from collections import namedtuple

Bounds = namedtuple('Bounds', ['lower', 'upper'])

UI_FILE = 'ui/rbfn.glade'
PLOT_POINTS = 100

xlim = Bounds(-5, 5)

class FunctionApprox():
    functions = [
        lambda x: ((x - 2) * (2*x + 1)) / (1 + x**2),
        lambda x: np.cos(np.pi / 2 * x),
        lambda x: 1.1 * (1 - x + 2 * x**2) * np.exp(-0.5 * x**2),
        None
    ]

    def __init__(self):
        self.rbfn = None
        self.function = None
        self.plotter = Plotter()

        # build UI
        self.builder = Gtk.Builder()
        self.builder.add_from_file(UI_FILE)
        self.builder.connect_signals(self)

        container = self.builder.get_object('figure_container')
        container.add(self.plotter.canvas)

        # save object references
        self.main_window = self.builder.get_object('main_window')
        self.hidden_layer_spin = self.builder.get_object('hidden_layer_spin')
        self.function_combo = self.builder.get_object('function_combo')
        self.samples_spin = self.builder.get_object('samples_spin')
        self.learning_rate_spin = self.builder.get_object('learning_rate_spin')
        self.max_epochs_spin = self.builder.get_object('max_epochs_spin')
        self.goal_error_spin = self.builder.get_object('goal_error_spin')

        self.set_functions()

    def show(self):
        self.main_window.show_all()
        Gtk.main()

    def set_functions(self):
        # connect signals
        for i, function in enumerate(self.functions):
            radio = self.builder.get_object('function_radio_{}'.format(i))
            radio.connect('toggled', self.on_function_toggled, function)

        # first radio button is active by default
        active = self.builder.get_object('function_radio_0')

        # emit the toggled signal for the active radio button, so the corresponding
        # function is plotted.
        active.emit('toggled')

    # --------------
    # Event handlers
    # --------------
    def on_main_window_delete_event(self, *args):
        Gtk.main_quit(*args)

    def on_function_toggled(self, button, function):
        if button.get_active():
            self.function = function
            self.clear_canvas()
            self.plot_function()

            if not function:
                self.plotter.sampling = True
            else:
                self.plotter.sampling = False

    def on_train_button_clicked(self, button):
        hidden_neurons = self.hidden_layer_spin.get_value_as_int()

        self.rbfn = RbfNetwork(1, hidden_neurons, 1)

        x, y = self.get_samples()
        x = x.reshape( (len(x), 1) )

        self.plotter.gauss_axes.lines = []

        self.rbfn.train(
            x, y,
            self.learning_rate_spin.get_value(),
            self.max_epochs_spin.get_value_as_int(),
            self.goal_error_spin.get_value(),
            self.on_epoch)

        self.plot_gaussians()
        self.plot_approximated()

    def get_samples(self):
        if self.function:
            samples = self.samples_spin.get_value_as_int()
            x = np.linspace(xlim.lower, xlim.upper, samples)
            y = self.function(x)

            self.plotter.plot_samples(x, y)

            return (x, y)
        else:
            # when no function is selected
            samples = self.plotter.samples_plt

            return samples.get_data()

    def on_epoch(self):
        self.plot_approximated()

        while Gtk.events_pending():
            Gtk.main_iteration()

    # ----------
    # Plotting
    # ----------
    def clear_canvas(self):
        self.plotter.plot_function([], [])
        self.plotter.plot_approximated([], [])
        self.plotter.plot_samples([], [])

    def plot_approximated(self):
        x = np.linspace(xlim.lower, xlim.upper, PLOT_POINTS)
        y = np.empty(PLOT_POINTS)

        inputs = x.reshape((PLOT_POINTS, 1))

        for i, vector in enumerate(inputs):
            y[i], = self.rbfn.test(vector)

        self.plotter.plot_approximated(x, y)

    def plot_function(self):
        if self.function:
            x = np.linspace(xlim.lower, xlim.upper, PLOT_POINTS)
            y = self.function(x)

            self.plotter.plot_function(x, y)

    def plot_gaussians(self):
        xdata = np.linspace(xlim.lower, xlim.upper, PLOT_POINTS)
        ydata = np.empty(PLOT_POINTS)

        self.plotter.gauss_axes.lines = []

        for neuron in self.rbfn._rbf_neurons:
            for i, x in enumerate(xdata):
                ydata[i] = neuron.gaussian(x)

            self.plotter.gauss_axes.plot(xdata, ydata)

        self.plotter.canvas.draw()

if __name__ == '__main__':
    FunctionApprox().show()
