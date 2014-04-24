__author__ = 'elubin'

import matplotlib.pyplot as plt
import numpy

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

    plot_data(data, x_label, x_range, "Proportion of Population", graph_options[GraphOptions.TITLE_KEY], num_strats, graph_options=graph_options)


def plot_single_data_set(data, x_label, x_values, y_label, title, num_categories, graph_options=None):
    legend_labels = graph_options[GraphOptions.LEGEND_LABELS_KEY]
    graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda i, j: legend_labels(j)
    plot_data([data], x_label, x_values, y_label, lambda i: title, [num_categories], graph_options=graph_options)


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

    old_options = GraphOptions.default.copy()
    if graph_options is not None:
        old_options.update(graph_options)
    graph_options = old_options

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