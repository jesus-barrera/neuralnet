 # -*- coding: utf-8 -*-
import numpy as np
from gi.repository import Gtk
from gi.repository.GdkPixbuf import Pixbuf

from neuralnet.som import SelfOrganizingMap
from plotting.som import Plotter
from collections import namedtuple

Bounds = namedtuple('Bounds', ['lower', 'upper'])

UI_FILE = 'ui/som.glade'

xlim = Bounds(-5, 5)
ylim = Bounds(-5, 5)

class SomApp(Gtk.Builder):
    def __init__(self):
        Gtk.Builder.__init__(self)

        self.som = None
        self.plotter = Plotter(xlim, ylim)


        # build UI
        self.add_from_file(UI_FILE)
        self.connect_signals(self)

        container = self.get_object('figure_container')
        container.add(self.plotter.canvas)

    def show(self):
        self.get_object('main_window').show_all()
        Gtk.main()

    # --------------
    # Event handlers
    # --------------
    def on_main_window_delete_event(self, *args):
        Gtk.main_quit(*args)

    def on_train_button_clicked(self, button):
        rows = self.get_object('rows_spin').get_value_as_int()
        cols = self.get_object('cols_spin').get_value_as_int()

        self.som = SelfOrganizingMap(2, rows, cols)

        self.som.train(
            self.get_training_set(),
            self.get_object('max_epochs_spin').get_value_as_int(),
            self.get_object('neighborhood_spin').get_value(),
            self.get_object('learning_rate_spin').get_value(),
            self.on_epoch)

        print 'Done!'

    def on_sampling_button_clicked(self, button):
        samples = self.get_object('samples_spin').get_value_as_int()

        xdata = np.random.uniform(xlim.lower, xlim.upper, samples)
        ydata = np.random.uniform(ylim.lower, ylim.upper, samples)

        self.plotter.add_samples(xdata, ydata)

    def on_clear_button_clicked(self, button):
        self.plotter.clear_canvas()

    def get_training_set(self):
        xdata, ydata = self.plotter.samples_plt.get_data()

        xdata = xdata.reshape((xdata.size, 1))
        ydata = ydata.reshape((ydata.size, 1))

        return np.concatenate((xdata, ydata), axis=1)

    def on_epoch(self):
        self.plotter.plot_weights(self.som)

        while Gtk.events_pending():
            Gtk.main_iteration()

    # ----------
    # Plotting
    # ----------


if __name__ == '__main__':
    SomApp().show()
