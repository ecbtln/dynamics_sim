__author__ = 'elubin'



# architecture to persist results of a single simulation across many runs


class SingleSimulationOutcome(object):
    def __init__(self, dyn_class, dyn_init_kwargs, game_class, game_kwargs, results):
        pass



    def last_generation(self):
        pass

    # redo the simulation and get more results
    def redo(self):
        pass

    def save(self):
        return "PATH_TO_FILE"

# a class that handles simulation results
# all simulations must be from the same dynamics, with the

class VariedParameterSimulation(object):
    def __init__(self, game):
        self. game = game # the game being played (a pointer to the class)
        # keep each simulation result as an element in an array
        # each array is a part of the dictionary that is keyed by a tuple of tuples defining the parameters for a given
        # fixed simulation. Those parameters are sorted alphabetically to ensure consistency in keys

    @classmethod
    def load_from_file(cls, path_to_file):
        pass

    def persist(self):
        return "PATH_TO_FILE"

    # redo the simulation(s), and
    def redo(self):
        pass


