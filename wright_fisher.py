__author__ = 'eblubin@mit.edu'
import math
import numpy as np
from matplotlib import pyplot as plt
import random
import operator
import heapq

# The precision of the decimal comparison operations this should not need any changing
DECIMAL_PRECISION = 5

# Colors used to plot the senders and receivers
GRAPH_COLORS = 'mcrgbyk'



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




class WrightFisher(object):
    def __init__(self, payoff_matrix, player_dist, pop_size=100, fitness_func=lambda p, w: math.e**(p*w),  mu=0.05, ss=0.8,
                  tolerance=0.08):
        # TODO: verify that "depth" of payoff matrix matches number of elements in player_dist
        self.num_player_types = len(player_dist)

        self.num_strats = []
        root = self.payoff_matrix
        for i in range(self.num_player_types):
            root = root[0]
            self.num_strats.append(len(root))

        self.mu = mu
        self.fitness_func = lambda payoff: fitness_func(payoff, ss)
        self.num_players = round_individuals([pop_size * x for x in player_dist])

        self.payoff_matrix = payoff_matrix

    def get_payoff(self, recipient, *strats):
        """
        Get the payoff for the player index recipient, by specifiying the strategies that everyone plays in increasing
        player order
        """
        matrix = self.payoff_matrix[recipient]
        for idx in strats:
            matrix = matrix[idx]
        return matrix


    def simulate(self, num_gens=100, debug_state=None):

        if debug_state is not None:
            state = debug_state
        else:
            state = []
            for n_p, n_s in zip(self.num_players, self.num_strats):
                frequencies = [0 for x in range(n_s)]
                for i in xrange(n_p):
                    frequencies[random.randint(0, n_s-1)] += 1
                state.append(frequencies)


        strategies = [np.zeros((num_gens, x)) for x in self.num_strats]

        # record initial state
        for i, x in enumerate(state):
            strategies[i][0, :] = x

        for gen in xrange(num_gens):
            # Calculate expected payoffs each player gets by playing a particular strategy based on the current state
            payoff = [[self.get_expected_payoff(p_idx, s_idx, state)
                       for s_idx in range(num_strats_i)]
                      for p_idx, num_strats_i in enumerate(self.num_strats)]


            # Calculate fitness for each individual in the population
            fitness = [[self.fitness_func(p) for p in j] for j in payoff]


            # Generate offspring population probabilistically based on
            # fitness/avg_fitness

            # Choose which offpsring will mutate
            # Randomly generate 1s with probability mu
            # Use "find" to find the 1s.  Calculate the length of the array
            # find returns and use this to (a) generate random strategies,
            # and (b) swap those strategies into the vector of offspring
            # record state
            for i, x in enumerate(state):
                strategies[i][gen + 1, :] = x


    def get_expected_payoff(self, player_idx, strategy, current_state):
        return self._iterate_through_players(player_idx, 0, {player_idx: strategy}, 1.0, current_state)


    def _iterate_through_players(self, target_player_idx, current_player_idx, other_player_stategies, probability, current_state):
        if len(other_player_stategies) == self.num_player_types:
            strats = [0] * self.num_player_types
            for i in range(self.num_player_types):
                strats[i] = other_player_stategies[i]

            payoff = self.get_payoff(target_player_idx, *strats)
            return payoff * probability

        elif current_player_idx in other_player_stategies:
            # skip it, we already picked the strategy
            return self._iterate_through_players(target_player_idx, current_player_idx + 1, other_player_stategies, probability, current_state)
        else:
            # iterate over the current player idx dimension, recursively calling yourself on every iteration'
            payoff = 0

            for strat in range(self.num_strats[current_player_idx]):
                n = current_state[current_player_idx][strat]
                p = float(n) / self.num_players[current_player_idx]
                dict_copy = dict(other_player_stategies.items())
                dict_copy[current_player_idx] = strat
                payoff += self._iterate_through_players(target_player_idx, current_player_idx + 1, dict_copy, probability * p, current_state)

            return payoff











