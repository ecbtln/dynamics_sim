__author__ = 'eblubin@mit.edu'
import math
import numpy as np
from dynamics import DynamicsSim, round_individuals

# Colors used to plot the players
GRAPH_COLORS = 'mcrgbyk'


class WrightFisher(DynamicsSim):
    def __init__(self, fitness_func=lambda p, w: math.e**(p*w),  mu=0.05, selection_strength=0.8, *args, **kwargs):
        super(WrightFisher, self).__init__(*args, **kwargs)
        self.mu = mu
        self.fitness_func = lambda payoff: fitness_func(payoff, selection_strength)

    def next_generation(self, previous_state):
        state = []
        # Calculate expected payoffs each player gets by playing a particular strategy based on the current state
        payoff = [[self.get_expected_payoff(p_idx, s_idx, previous_state)
                   for s_idx in range(num_strats_i)]
                  for p_idx, num_strats_i in enumerate(self.num_strats)]

        # Calculate fitness for each individual in the population (based on what strategy they are playing)
        fitness = [[self.fitness_func(p) for p in j] for j in payoff]

        # Generate offspring population probabilistically based on
        # fitness/avg_fitness, with some potential for each individual to be mutated
        for player_idx, (strategy_distribution, fitnesses, num_players) in enumerate(zip(previous_state, fitness, self.num_players)):
            num_strats = len(strategy_distribution)
            total_mutations = 0
            new_player_state = np.zeros(num_strats)
            for strategy_idx, n in enumerate(strategy_distribution):
                f = fitness[player_idx][strategy_idx]

                # sample from binomial distribution to get number of mutations for strategy
                mutations = np.random.binomial(n, self.mu)
                n -= mutations
                total_mutations += mutations
                new_player_state[strategy_idx] = n * f
                # distribute player strategies proportional n * f
                # don't use multinomial, because that adds randomness we don't want yet
            new_player_state *= float(num_players - total_mutations) / new_player_state.sum()
            new_player_state = np.array(round_individuals(new_player_state))

            new_player_state += np.random.multinomial(total_mutations, [1. / num_strats] * num_strats)
            state.append(new_player_state)

        return state

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











