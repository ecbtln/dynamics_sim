__author__ = 'elubin'
from abc import ABCMeta, abstractmethod
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt


# DEFAULT GRAPH OPTIONS, can be overridden
GRAPH_COLORS = "bgrcmykw"
GRAPH_X_LABEL = "Generation #"
GRAPH_Y_LABEL = "Proportion of Population"
GRAPH_LEGEND_LOCATION = "center right"
GRAPH_SHOW_GRID = True
GRAPH_TITLE_FORMAT = "Population Dynamics for Player %d"

# The precision of the decimal comparison operations this should not need any changing
DECIMAL_PRECISION = 5


class DynamicsSim(object):
    __metaclass__ = ABCMeta

    def __init__(self, payoff_matrix, player_frequencies, strategy_labels=None, pop_size=100):
        """ set pop_size equal to 0 to use infinite players (where we only care about their relative frequencies) """

        assert math.fsum(player_frequencies) == 1.0

        assert pop_size >= 0

        if pop_size > 0:
            self.num_players = round_individuals([pop_size * x for x in player_frequencies])
            assert sum(self.num_players) == pop_size
            self.infinite_pop_size = False
        else:
            self.num_players = player_frequencies
            self.infinite_pop_size = True

        self.payoff_matrix = payoff_matrix
        self.num_player_types = len(player_frequencies)

        self.num_strats = []
        root = self.payoff_matrix[0]
        for i in range(self.num_player_types):
            self.num_strats.append(len(root))
            root = root[0]

        self.verify_payoff_matrix_dimensions()

        if strategy_labels is None:
            self.labels = [["X_%d" % x for x in range(strats)] for strats in self.num_strats]
        else:
            assert len(strategy_labels) == self.num_player_types
            for p_labels, num_strats in zip(strategy_labels, self.num_strats):
                assert len(p_labels) == num_strats
                for label in p_labels:
                    assert isinstance(label, (str, basestring))
            self.labels = strategy_labels

    def verify_payoff_matrix_dimensions(self):
        # verify that "depth" of each payoff matrix matches number of elements in player_dist
        for m in self.payoff_matrix:
            self._verify_dimensions(m, self.num_strats[:])

    def _verify_dimensions(self, m, num_strats):
        if len(num_strats) == 0:
            assert isinstance(m, (int, float))
            return
        n = num_strats.pop(0)
        assert n == len(m)
        for i in m:
            self._verify_dimensions(i, num_strats[:])

    def get_payoff(self, recipient, *strats):
        """
        Get the payoff for the player index recipient, by specifiying the strategies that everyone plays in increasing
        player order

        """
        matrix = self.payoff_matrix[recipient]
        for idx in strats:
            matrix = matrix[idx]
        return matrix

    @abstractmethod
    def next_generation(self, previous):
        return []

    def validate_state(self, s):
        """
        Verify validity of state, each state is an array of numpy arrays, one for every player type
        Also needs to coerce any arrays to numpy arrays
        """
        assert len(s) == self.num_player_types
        for i, (p, expected, n_strats) in enumerate(zip(s, self.num_players, self.num_strats)):
            if isinstance(p, (list, tuple)):
                p = np.ndarray(p)
                s[i] = p

            assert isinstance(p, np.ndarray)
            assert p.sum() == expected
            assert len(p) == n_strats

        return s

    def simulate(self, num_gens=100, debug_state=None, graph=False, graph_options=None):
        if debug_state is not None:
            state = self.validate_state(debug_state)
        else:
            if not self.infinite_pop_size:
                distribution_for_player = lambda n_p, n_s: np.random.multinomial(n_p, [1./n_s] * n_s)
            else:
                distribution_for_player = lambda n_p, n_s: np.random.dirichlet([1] * n_s) * n_p

            state = [distribution_for_player(n_p, n_s) for n_p, n_s in zip(self.num_players, self.num_strats)]

        strategies = [np.zeros((num_gens, x)) for x in self.num_strats]

        # record initial state
        for i, x in enumerate(state):
            strategies[i][0, :] = x

        for gen in xrange(num_gens):
            state = self.validate_state(self.next_generation(state))
            # record state
            for i, x in enumerate(state):
                strategies[i][gen + 1, :] = x

        if graph:
            if graph_options is None:
                graph_options = {}

            colors = graph_options.get('colors', GRAPH_COLORS)

            # graph the results
            for i, (player, labels) in enumerate(zip(strategies, self.labels)):
                plt.title(graph_options.get('title_format', GRAPH_TITLE_FORMAT) % i)

                # iterate over all the generations
                num_gens, num_strats = player.shape

                plt.xlim([0, num_gens + 2])
                plt.ylim([0, 1])
                plt.xlabel(graph_options.get('x_label', GRAPH_X_LABEL))
                plt.ylabel(graph_options.get('y_label', GRAPH_Y_LABEL))
                plt.grid(graph_options.get('grid', GRAPH_SHOW_GRID))

                for gen_i in range(num_gens):
                    # iterate over all the strategies
                    for strat_i in range(num_strats):
                        val = player[gen_i, strat_i]
                        if not self.infinite_pop_size:
                            # value is in whole number terms, scale to proportions here
                            pass
                        plt.scatter(gen_i, val, c=colors[strat_i % num_strats])

                plt.legend(labels, loc=graph_options.get('legend_location', GRAPH_LEGEND_LOCATION))
                plt.show()

        # return final state after simulation
        return [x[num_gens] for x in strategies]

    @staticmethod
    def round_individuals(unrounded_frequencies):
        """
        Due to integer cutoffs, the number of senders and receivers might not be consistent. This take the integer part
        of each of the inputs and then assign the remaining few leftovers (so that the sum is the sum of the original
        floats) in a way such that the numbers with higher decimal parts will get the extra int before those with lower.
        """
        unrounded_total = math.fsum(unrounded_frequencies)
        total = int(round(unrounded_total, DECIMAL_PRECISION))

        int_num_senders = [int(x) for x in unrounded_frequencies]

        diff = total - sum(int_num_senders)
        if diff > 0:
            # note the difference needs to be negative, because heapq's only implement a minimum priority queue but
            # we want max priority queue
            thresh = [((x - y), i) for i, (x, y) in enumerate(zip(int_num_senders, unrounded_frequencies))]
            heapq.heapify(thresh)
            while diff > 0:
                v, i = heapq.heappop(thresh)
                int_num_senders[i] += 1
                diff -= 1
        assert sum(int_num_senders) == total, "the total number of individuals after rounding must be the same as " \
                                              "before rounding"

        return int_num_senders



