from dynamics import StochasticDynamicsSimulator
import numpy


class Moran(StochasticDynamicsSimulator):
    """
    A stochastic dynamics simulator that performs the Moran process on all player types in the population.
    See U{Moran Process<http://en.wikipedia.org/wiki/Moran_process#Selection>}
    """
    def __init__(self, num_iterations_per_time_step=1, *args, **kwargs):
        """
        The constructor for the Moran dynamics process, that the number of births/deaths to process per time step.

        @param num_iterations_per_time_step: the number of iterations of the Moran process we do per time step
        @type num_iterations_per_time_step: int
        """
        super(Moran, self).__init__(*args, **kwargs)
        assert num_iterations_per_time_step >= 1
        self.num_iterations_per_time_step = num_iterations_per_time_step

    def next_generation(self, previous_state):
        next_state = []

        # copy to new state
        for p in previous_state:
            next_state.append(p.copy())

        fitness = self.calculate_fitnesses(next_state)

        minimum_total = min(p.sum() for p in next_state)
        # make sure there are enough individuals of each type to take away 2 * num_iterations_per_time_step
        num_iterations = min(self.num_iterations_per_time_step * 2, minimum_total) / 2

        for p, f in zip(next_state, fitness):
            reproduce = numpy.zeros(len(p))

            for i in range(num_iterations):
                # sample from distribution to determine winner and loser (he who reproduces, he who dies)
                weighted_total = sum(n_i * f_i for n_i, f_i in zip(p, f))
                dist = numpy.array([n_i * f_i / weighted_total for n_i, f_i in zip(p, f)])
                sample = numpy.random.multinomial(1, dist)
                p -= sample
                reproduce += sample

            for i in range(num_iterations):
                # now determine who dies from what's left
                total = p.sum()
                dist = [n_i / float(total) for n_i in p]
                p -= numpy.random.multinomial(1, dist)

            p += reproduce * 2


        return next_state