import math
from dynamics import StochasticDynamicsSimulator
import numpy


class Moran(StochasticDynamicsSimulator):
    """
    A stochastic dynamics simulator that performs the
    U{Moran Process<http://en.wikipedia.org/wiki/Moran_process#Selection>} on all player types in the population.
    """
    def __init__(self, num_iterations_per_time_step=1, *args, **kwargs):
        """
        The constructor for the Moran dynamics process, that the number of births/deaths to process per time step.

        @param num_iterations_per_time_step: the number of iterations of the Moran process we do per time step
        @type num_iterations_per_time_step: int
        """
        super(Moran, self).__init__(*args, hide_markers=True, **kwargs)
        assert num_iterations_per_time_step >= 1
        self.num_iterations_per_time_step = num_iterations_per_time_step

    def next_generation(self, previous_state):
        for i in range(self.num_iterations_per_time_step):
            next_state = []

            # copy to new state
            for p in previous_state:
                next_state.append(p.copy())

            fitness = self.calculate_fitnesses(next_state)

            for p, f in zip(next_state, fitness):
                # sample from distribution to determine winner and loser (he who reproduces, he who dies)
                weighted_total = sum(n_i * f_i for n_i, f_i in zip(p, f))
                total = p.sum()
                reproduce = numpy.random.multinomial(1, numpy.array([n_i * f_i / weighted_total for n_i, f_i in zip(p, f)]))

                # temporarily take away the reproducer
                p -= reproduce

                # now determine who dies from what's left
                weighted_total = sum(n_i * f_i for n_i, f_i in zip(p, f))
                death = numpy.random.multinomial(1, numpy.array([n_i * f_i / weighted_total for n_i, f_i in zip(p, f)]))

                p += reproduce * 2 - death

            previous_state = next_state

        return previous_state