__author__ = 'elubin'
from abc import ABCMeta, abstractmethod
import heapq
import math
import numpy as np

# The precision of the decimal comparison operations this should not need any changing
DECIMAL_PRECISION = 5


class DynamicsSimulator(object):
    __metaclass__ = ABCMeta

    # TODO: see whether you care about every period or just end state. stochastic vs deterministic
    def __init__(self, payoff_matrix, player_frequencies=None, pop_size=100, stochastic=False, hide_markers=False):
        """ set pop_size equal to 0 to use infinite players (where we only care about their relative frequencies)
        @param hide_markers: whether or not to hide the markers for a graph and just show the lines, since
            Moran process, for example, tends to not have drastic spikes and is more smooth
        @type hide_markers: bool
        """

        assert math.fsum(player_frequencies) == 1.0

        assert pop_size >= 0

        if pop_size > 0:
            self.num_players = self.round_individuals([pop_size * x for x in player_frequencies])
            assert sum(self.num_players) == pop_size
            self.infinite_pop_size = False
        else:
            self.num_players = player_frequencies
            self.infinite_pop_size = True
        self.pop_size = pop_size
        self.pm = payoff_matrix
        self.stochastic = stochastic
        self.hide_markers = hide_markers


    @abstractmethod
    def next_generation(self, previous):
        return []

    def validate_state(self, s):
        """
        Verify validity of state, each state is an array of numpy arrays, one for every player type
        Also needs to coerce any arrays to numpy arrays
        """
        assert len(s) == self.pm.num_player_types
        for i, (p, expected, n_strats) in enumerate(zip(s, self.num_players, self.pm.num_strats)):
            if isinstance(p, (list, tuple)):
                p = np.array(p)
                s[i] = p

            assert isinstance(p, np.ndarray)
            assert p.sum() == expected
            assert len(p) == n_strats

        return s

    def simulate(self, num_gens=100, debug_state=None):
        if debug_state is not None:
            state = self.validate_state(debug_state)
        else:
            if not self.infinite_pop_size:
                distribution_for_player = lambda n_p, n_s: np.random.multinomial(n_p, [1./n_s] * n_s)
            else:
                distribution_for_player = lambda n_p, n_s: np.random.dirichlet([1] * n_s) * n_p

            state = [distribution_for_player(n_p, n_s) for n_p, n_s in zip(self.num_players, self.pm.num_strats)]

        strategies = [np.zeros((num_gens, x)) for x in self.pm.num_strats]

        # record initial state
        for i, x in enumerate(state):
            strategies[i][0, :] = x

        for gen in xrange(num_gens - 1):
            state = self.validate_state(self.next_generation(state))
            # record state
            for i, x in enumerate(state):
                strategies[i][gen + 1, :] = x

        return strategies

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


class StochasticDynamicsSimulator(DynamicsSimulator):
    __metaclass__ = ABCMeta

    def __init__(self, fitness_func=lambda p, w: math.e**(p*w),  selection_strength=0.8,  *args, **kwargs):
        """
        The constructor for stochastic dynamics processes. They all reproduce based on relative fitness, so this
        abstract class provides a convenient fitness calculate function based on the current state using the provided
        fitness_func.

        @param fitness_func:
        @type fitness_func: lambda that takes in two arguments, the payoff and the selection strength, and returns the fitness
        @param selection_strength: the selection strength that will be used in the fitness function
        @type selection_strength: float
        """
        super(StochasticDynamicsSimulator, self).__init__(*args, stochastic=True, **kwargs)
        self.fitness_func = lambda payoff: float(fitness_func(payoff, selection_strength))


    @abstractmethod
    def next_generation(self, previous):
        return []

    def calculate_fitnesses(self, state):
        # Calculate expected payoffs each player gets by playing a particular strategy based on the current state
        payoff = [[self.pm.get_expected_payoff(p_idx, s_idx, state)
                   for s_idx in range(num_strats_i)]
                  for p_idx, num_strats_i in enumerate(self.pm.num_strats)]

        # Calculate fitness for each individual in the population (based on what strategy they are playing)
        return [[self.fitness_func(p) for p in j] for j in payoff]
