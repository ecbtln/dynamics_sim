__author__ = 'elubin'
from abc import ABCMeta, abstractmethod
import heapq
import math
import numpy as np

# The precision of the decimal comparison operations this should not need any changing
DECIMAL_PRECISION = 5


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


class DynamicsSim(object):
    __metaclass__ = ABCMeta

    def __init__(self, payoff_matrix, player_frequencies, pop_size=100):

        self.num_players = round_individuals([pop_size * x for x in player_frequencies])

        self.payoff_matrix = payoff_matrix
        self.num_player_types = len(player_frequencies)

        self.num_strats = []
        root = self.payoff_matrix[0]
        for i in range(self.num_player_types):
            self.num_strats.append(len(root))
            root = root[0]

        self.verify_payoff_matrix_dimensions()

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

    def simulate(self, num_gens=100, debug_state=None, graph_options=None):
        if debug_state is not None:
            state = debug_state
        else:
            state = [np.random.multinomial(n_p, [1./n_s] * n_s) for n_p, n_s in zip(self.num_players, self.num_strats)]

        strategies = [np.zeros((num_gens, x)) for x in self.num_strats]

        # record initial state
        for i, x in enumerate(state):
            strategies[i][0, :] = x

        for gen in xrange(num_gens):
            # TODO: verify dimensions of output array of consistent
            state = self.next_generation(state)

            # record state
            for i, x in enumerate(state):
                strategies[i][gen + 1, :] = x




