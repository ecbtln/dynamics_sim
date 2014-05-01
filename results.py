__author__ = 'elubin'
from plot import GraphOptions, plot_single_data_set, plot_3d_data_set
import numpy

# TODO: architecture to persist results of a single simulation across many runs


# class SingleSimulationOutcome(object):
#     def __init__(self, dyn_class, dyn_init_kwargs, game_class, game_kwargs, results):
#         pass
#
#
#
#     def last_generation(self):
#         pass
#
#     # redo the simulation and get more results
#     def redo(self):
#         pass
#
#     def save(self):
#         return "PATH_TO_FILE"
#
# # a class that handles simulation results
# # all simulations must be from the same dynamics, with the
#
# class VariedParameterSimulation(object):
#     def __init__(self, game):
#         self. game = game # the game being played (a pointer to the class)
#         # list of lists, N deep
#         # list of independent parameters, of length N, detailing on which level each parameter that is varied, and its range
#
#         # keep each simulation result as an element in an array
#         # each array is a part of the dictionary that is keyed by a tuple of tuples defining the parameters for a given
#         # fixed simulation. Those parameters are sorted alphabetically to ensure consistency in keys
#
#     @classmethod
#     def load_from_file(cls, path_to_file):
#         pass
#
#     def persist(self):
#         return "PATH_TO_FILE"
#
#     # redo the simulation(s), and
#     def redo(self):
#         pass
#


class NDimensionalData(object):
    """
    A helper class that encapsulates varied multi-dimensional results. The results are stored as a recursive
    list of lists, whereby to get the simulation results for a given fixed set of parameters, we can iterate through
    the recursive list and index based on the indices of the independent parameter values desired.
    """
    def __init__(self, data, independent_parameters):
        """
        Initialize the multi-dimensional data container with the given data.
        @param data: A recursive list of lists representing the data in a tree-like form, where the address to the data
            corresponds to the index of the parameter's value that the data takes, in order of the parameters in the list
        @type data: list(...), of length exactly len(independent_parameters)
        @param independent_parameters: a list of the IndependentParameter instances
        @type independent_parameters: list(L{VerboseIndependentParameter})
        """
        self.independent_parameters = independent_parameters
        self.data = data

        self._validate_data(data, 0)

    def _validate_data(self, data, idx):
        if idx == len(self.independent_parameters):
            return

        assert len(data) == len(self.independent_parameters[idx])

        for x in data:
            self._validate_data(x, idx + 1)

    def num_dimensions(self):
        """
        Get the number of dimensions that this data represents (one more than the number of independent variables)
        """
        return len(self.independent_parameters) + 1

    def graph(self, equilibria):
        """
        Graph the data, if possible. Subclasses may override this method.

        @param equilibria: the list of available equlibria, as strings
        @type equilibria: list
        """
        raise ValueError("Cannot graph %d dimensions, marginalize the data and try again")

    def marginalize(self, **kwargs):
        """
        Convert the N dimensional data to a lower-order dimension by iterating through the data and stitching
        together the appropriate subarrays. The keyword arguments are keyed by the keyword of the parameter, and their
        value is the value on which to marginalize the data (by fixing that parameter to that value, and therefore
        looking at one cross section of the higher order data
        @return: A new instance of NDimensionData or any of its subclasses
        @rtype: NDimensionalData
        """
        # TODO: still needs to be implemented
        pass

    @classmethod
    def initialize(cls, data, independent_parameters):
        """
        A helper class method that initializes the class (or subclass) of L{NDimensionalData} with the given keyword
        arguments. It first checks the number of parameters to vary, and then chooses the appropriate class accordingly.
        @param data:
        @type data:
        @param independent_parameters:
        @type independent_parameters:
        """
        n = len(independent_parameters)
        constructor = cls
        if n == 1:
            constructor = TwoDimensionalData
        elif n == 2:
            constructor = ThreeDimensionalData

        return constructor(data, independent_parameters)


class TwoDimensionalData(NDimensionalData):
    """
    A subclass that aids in the graphing of two dimension data (varying one parameter across a range of values)
    """
    def graph(self, equilibria):
        graph_options = {}
        graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda i: equilibria[i]
        x = self.independent_parameters[0]
        x_values = list(x)
        n_equilibria = len(equilibria)
        data = numpy.array(self.data)
        plot_single_data_set(data, x.key, x_values,
                             "Equilibrium Proportion",
                             "%s:(%.3f ... %.3f)" % (x.key, x.lb, x.ub),
                             n_equilibria, graph_options)


class ThreeDimensionalData(NDimensionalData):
    """
    A subclass that aids in the graphing of three dimensional data (varying two parameters)
    """
    def graph(self, equilibria):
        graph_options = {}
        graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda i: equilibria[i]
        x = self.independent_parameters[0]
        y = self.independent_parameters[1]
        x_values = list(x)
        y_values = list(y)
        nx = len(x_values)
        ny = len(y_values)
        n_equilibria = len(equilibria)
        data = numpy.zeros((nx, ny, n_equilibria))
        for i in range(nx):
            for j in range(ny):
                data[i, j, :] = self.data[i][j]
        plot_3d_data_set(data, x.key, x_values, y.key, y_values,
                         "Equilibrium Proportion", "%s:(%.3f ... %.3f), %s:(%.3f ... %.3f) " % (x.key,
                                                                                                x.lb,
                                                                                                x.ub,
                                                                                                y.key,
                                                                                                y.lb,
                                                                                                y.ub),
                         n_equilibria, graph_options)



