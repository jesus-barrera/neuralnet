 # -*- coding: utf-8 -*-
import time
import numpy as np
from gi.repository import Gtk
from neuralnet.mlp import MultiLayerPerceptron as MLP
from plot import Plotter

UI_FILE = 'ui.glade'

class Classifier:
    def __init__(self):
        self.mlp = None
        self.plotter = Plotter()

        # build UI
        self.builder = Gtk.Builder()
        self.builder.add_from_file(UI_FILE)
        self.builder.connect_signals(self)

        container = self.builder.get_object('figure_container')
        container.add(self.plotter.canvas)

        # save object references
        self.main_window = self.builder.get_object('main_window')
        self.progressbar = self.builder.get_object('progressbar')
        self.learning_rate_spin = self.builder.get_object('learning_rate_spin')
        self.error_goal_spin = self.builder.get_object('error_goal_spin')
        self.max_epochs_spin = self.builder.get_object('max_epochs_spin')
        self.hidden_layers_entry = self.builder.get_object('hidden_layers_entry')
        self.x_spin = self.builder.get_object('x_spin')
        self.y_spin = self.builder.get_object('y_spin')

    def show(self):
        self.main_window.show_all()
        Gtk.main()

    # --------------
    # Event handlers
    # --------------
    def on_main_window_delete_event(self, *args):
        Gtk.main_quit(*args)

    def on_train_button_clicked(self, button):
        self.max_epochs = self.max_epochs_spin.get_value_as_int()

        self.plotter.clear_error()
        self.plotter.clear_separating_surface()
        self.plotter.error_axes.set_xlim(0, self.max_epochs)

        self.mlp = self.create_mlp()

        converged, epochs = self.mlp.train(
                self.plotter.training_set,
                self.learning_rate_spin.get_value(),
                self.max_epochs,
                self.error_goal_spin.get_value(),
                self.update)

        if not converged:
            print u'El perceptrón no convergió!'
        else:
            print u'El perceptrón convergió en {} épocas'.format(epochs)

        self.plotter.plot_separating_surface(self.mlp)

    def on_test_button_clicked(self, button):
        x = self.x_spin.get_value()
        y = self.y_spin.get_value()

        inputs = np.array([x, y])
        output, = self.mlp.test(inputs)

        self.plotter.plot_point(x, y, output)

    def on_restart_button_clicked(self, button):
        self.mlp = None
        self.update_progressbar(0)
        self.plotter.clear()

    def create_mlp(self):
        hidden_layers = self.hidden_layers_entry.get_text()
        hidden_layers = map(int, hidden_layers.split(','))

        shape = [2] + hidden_layers + [1] # two inputs / one output

        return MLP(shape)

    def update(self, epoch, error):
        self.plotter.update_error_line(error)
        self.update_progressbar(epoch)

        while Gtk.events_pending():
            Gtk.main_iteration()

    def update_progressbar(self, epoch):
        fraction = float(epoch) / self.max_epochs

        self.progressbar.set_fraction(fraction)
        self.progressbar.set_text(u'Época: {}'.format(epoch))

if __name__ == '__main__':
    Classifier().show()
