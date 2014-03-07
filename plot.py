__author__ = 'elubin'

import matplotlib.pyplot as plt

class GraphOptions:
    COLORS_KEY = 'colors'
    Y_LABEL_KEY = 'y_label'
    LEGEND_LOCATION_KEY = 'legend_location'
    SHOW_GRID_KEY = 'grid'
    TITLE_FORMAT_KEY = 'title_format'

    default = {COLORS_KEY: "bgrcmykw",
               Y_LABEL_KEY: "Proportion of Population",
               LEGEND_LOCATION_KEY: 'center right',
               SHOW_GRID_KEY: True,
               TITLE_FORMAT_KEY: "Population Dynamics for Player %d"}


def plot_data_for_players(data, x_range, x_label, num_strats, strategy_labels=lambda player_i, strat_i: "X_%d,%d" % (player_i, strat_i), num_players=None, graph_options=None):
    # data is a list of n = (the number of player types) of 2D arrays
    # 1st dimension indices are the index into the x_range array
    # 2nd dimension indices are the index of the strategy number
    # num_players tells us the total number of players devoted to each player type, or None if already normalized


    n_x = len(x_range)
    for player_state, n_strats in zip(data, num_strats):
        assert player_state.shape == n_x, n_strats

    if num_players is not None:
        assert len(num_players) == len(num_strats)

    old_options = GraphOptions.default.copy()
    if graph_options is not None:
        old_options.update(graph_options)
    graph_options = old_options

    colors = graph_options[GraphOptions.COLORS_KEY]

    # graph the results
    for i, player in enumerate(data):
        plt.title(graph_options[GraphOptions.TITLE_FORMAT_KEY] % i)

        # iterate over all the generations
        num_gens, num_strats = player.shape

        plt.xlim([0, num_gens + 2])
        plt.ylim([0, 1])
        plt.ylabel(graph_options[GraphOptions.Y_LABEL_KEY])
        plt.xlabel(x_label)
        plt.grid(graph_options[GraphOptions.SHOW_GRID_KEY])

        # TODO: change to plot an entire simulation all at once, rather than one point at a time
        # plt.plot(x_axis, avg_costs[: t, sender_type_idx], colors[sender_type_idx], label=lbl)
        for gen_i in range(num_gens):
            # iterate over all the strategies
            for strat_i in range(num_strats):
                val = player[gen_i, strat_i]
                if num_players is not None:
                    # value is in whole number terms, scale to proportions here
                    val /= float(num_players[i])
                plt.scatter(gen_i, val, c=colors[strat_i % num_strats])

        labels = [strategy_labels(i, j) for j in num_strats]
        plt.legend(labels, loc=graph_options[GraphOptions.LEGEND_LOCATION_KEY])
        plt.show()