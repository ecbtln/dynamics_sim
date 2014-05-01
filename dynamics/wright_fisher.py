__author__ = 'eblubin@mit.edu'
import numpy as np
from dynamics import StochasticDynamicsSimulator


class WrightFisher(StochasticDynamicsSimulator):
    def __init__(self, mu=0.05, *args, **kwargs):
        # TODO: don't allow pop_size of 0, wright fisher only works with finite pop size
        super(WrightFisher, self).__init__(*args, **kwargs)
        self.mu = mu

    def next_generation(self, previous_state):
        state = []

        fitness = self.calculate_fitnesses(previous_state)

        # Generate offspring population probabilistically based on
        # fitness/avg_fitness, with some potential for each individual to be mutated
        for player_idx, (strategy_distribution, fitnesses, num_players) in enumerate(zip(previous_state, fitness, self.num_players)):
            num_strats = len(strategy_distribution)
            total_mutations = 0
            new_player_state = np.zeros(num_strats)
            for strategy_idx, n in enumerate(strategy_distribution):
                f = fitness[player_idx][strategy_idx]

                # sample from binomial distribution to get number of mutations for strategy
                if n == 0:
                    mutations = 0
                else:
                    mutations = np.random.binomial(n, self.mu)
                n -= mutations
                total_mutations += mutations
                new_player_state[strategy_idx] = n * f
                # distribute player strategies proportional n * f
                # don't use multinomial, because that adds randomness we don't want yet
            new_player_state *= float(num_players - total_mutations) / new_player_state.sum()
            new_player_state = np.array(self.round_individuals(new_player_state))

            new_player_state += np.random.multinomial(total_mutations, [1. / num_strats] * num_strats)
            state.append(new_player_state)

        return state










