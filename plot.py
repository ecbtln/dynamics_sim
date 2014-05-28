__author__ = 'elubin'

import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D

class GraphOptions:
    COLORS_KEY = 'colors'
    Y_LABEL_KEY = 'y_label'
    LEGEND_LOCATION_KEY = 'legend_location'
    SHOW_GRID_KEY = 'grid'
    LEGEND_LABELS_KEY = 'legend_labels'
    TITLE_KEY = 'title'
    MARKERS_KEY = 'markers'
    NO_MARKERS_KEY = 'hide_markers'

    default = {COLORS_KEY: "bgrmykwc",
               MARKERS_KEY: "o.v8sh+xD|_ ",
               NO_MARKERS_KEY: False,
               Y_LABEL_KEY: "Proportion of Population",
               LEGEND_LOCATION_KEY: 'center right',
               SHOW_GRID_KEY: True,
               TITLE_KEY: lambda player_i: "Population Dynamics for Player %d" % player_i,
               LEGEND_LABELS_KEY: lambda graph_i, cat_i: "X_%d,%d" % (graph_i, cat_i)}


def plot_data_for_players(data, x_range, x_label, num_strats, num_players=None, graph_options=None):
    # data is a list of n = (the number of player types) of 2D arrays
    # 1st dimension indices are the index into the x_range array
    # 2nd dimension indices are the index of the strategy number
    # num_players tells us the total number of players devoted to each player type, or None if already normalized

    # normalize data, if needed
    if num_players is not None:
        normalized_data = []
        for i, player in enumerate(data):
            num_gens, n_strats = player.shape
            d = numpy.zeros((num_gens, n_strats))
            for gen_i in range(num_gens):
                for strat_i in range(n_strats):
                    d[gen_i, strat_i] = player[gen_i, strat_i] / float(num_players[i])
            normalized_data.append(d)
        data = normalized_data
    old_options = GraphOptions.default.copy()
    if graph_options is not None:
        old_options.update(graph_options)
    graph_options = old_options

    plot_data(data, x_label, x_range, "Proportion of Population", graph_options[GraphOptions.TITLE_KEY], num_strats, graph_options=graph_options)


def plot_single_data_set(data, x_label, x_values, y_label, title, num_categories, graph_options=None):
    legend_labels = graph_options[GraphOptions.LEGEND_LABELS_KEY]
    graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda i, j: legend_labels(j)
    plot_data([data], x_label, x_values, y_label, lambda i: title, [num_categories], graph_options=graph_options)


def _append_options(options):
    old_options = GraphOptions.default.copy()
    if options is not None:
        old_options.update(options)
    return old_options


def plot_data(data, x_label, x_values, y_label, title_i, num_categories, graph_options=None):
    """
    support for multiple 2d arrays, each as an entry in the data array
    All data should be normalized before being passed in
    """

    n_x = len(x_values)
    for state_i, n_cats in zip(data, num_categories):
        n, s = state_i.shape
        assert n == n_x
        assert s == n_cats

    graph_options = _append_options(graph_options)

    colors = graph_options[GraphOptions.COLORS_KEY]
    category_labels = graph_options[GraphOptions.LEGEND_LABELS_KEY]
    markers = graph_options[GraphOptions.MARKERS_KEY]

    # graph the results
    for i, data_i in enumerate(data):
        plt.figure(i)
        plt.title(title_i(i))

        # iterate over all the generations
        num_xs, n_cats = data_i.shape

        plt.xlim([x_values[0], x_values[-1]])
        plt.ylim([0, 1])
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.grid(graph_options[GraphOptions.SHOW_GRID_KEY])

        # iterate over all the categories
        for cat_i in range(n_cats):
            if graph_options[GraphOptions.NO_MARKERS_KEY]:
                marker = ' '
            else:
                marker = markers[cat_i / n_cats]
            plt.plot(x_values, data_i[:, cat_i], c=colors[cat_i % n_cats], lw=2, marker=marker)

        labels = [category_labels(i, j) for j in range(n_cats)]
        plt.legend(labels, loc=graph_options[GraphOptions.LEGEND_LOCATION_KEY])
    plt.show()


def plot_3d_data_set(data, x_label, x_values, y_label, y_values, z_label, title, num_categories, graph_options=None):

    graph_options = _append_options(graph_options)
    colors = graph_options[GraphOptions.COLORS_KEY]
    category_labels = graph_options[GraphOptions.LEGEND_LABELS_KEY]
    plt.close('all')
    fig = plt.figure()

    # iterate over all the generations
    num_xs, num_ys, n_cats = data.shape
    assert num_categories == n_cats

    # iterate over all the categories
    x_values = numpy.array(x_values)
    y_values = numpy.array(y_values)
    nx = len(x_values)
    ny = len(y_values)
    assert nx == num_xs
    assert ny == num_ys
    xs = numpy.repeat(x_values, ny)
    xs.resize((nx, ny))
    ys = numpy.tile(y_values, nx)
    ys.resize((nx, ny))

    dim = int(numpy.ceil(numpy.sqrt(n_cats)))
    for cat_i in range(n_cats):
        # print cat_i
        ax = fig.add_subplot(dim, dim, cat_i + 1, projection='3d')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        zs = data[:, :, cat_i]
        ax.set_title(category_labels(cat_i))
        ax.plot_wireframe(xs, ys, zs, color=colors[cat_i % n_cats])

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    # plot_wireframe
    # TODO: plot surface seems to look better, except it doesn't play nicely with multiple surfaces on the same grpah
    for cat_i in range(n_cats):
        zs = data[:, :, cat_i]
        ax.plot_wireframe(xs, ys, zs, color=colors[cat_i % n_cats])

    labels = [category_labels(j) for j in range(n_cats)]
    plt.legend(labels, loc=graph_options[GraphOptions.LEGEND_LOCATION_KEY])
    plt.show()