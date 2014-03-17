__author__ = 'eblubin@mit.edu'
import math
import numpy as np
from dynamics import StochasticDynamicsSimulator


class WrightFisher(StochasticDynamicsSimulator):
    def __init__(self, fitness_func=lambda p, w: math.e**(p*w),  mu=0.05, selection_strength=0.8, *args, **kwargs):
        # TODO: don't allow pop_size of 0, wright fisher only works with finite pop size
        super(WrightFisher, self).__init__(*args, **kwargs)
        self.mu = mu
        self.fitness_func = lambda payoff: fitness_func(payoff, selection_strength)

    def next_generation(self, previous_state):
        state = []
        # Calculate expected payoffs each player gets by playing a particular strategy based on the current state
        payoff = [[self.pm.get_expected_payoff(p_idx, s_idx, previous_state)
                   for s_idx in range(num_strats_i)]
                  for p_idx, num_strats_i in enumerate(self.pm.num_strats)]

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
            # TODO:
            new_player_state = np.array(self.round_individuals(new_player_state))

            new_player_state += np.random.multinomial(total_mutations, [1. / num_strats] * num_strats)
            state.append(new_player_state)

        return state










