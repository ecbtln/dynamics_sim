__author__ = 'elubin'

from dynamics_sim.payoff_matrix import PayoffMatrix


UNCLASSIFIED_EQUILIBRIUM = 'Unclassified'  #: the string used to identify an equilibrium that did not match any of the classification rules


class Game(object):
    """
    A class that is used to encapsulate the notion of a game theory game Each game is identified by a set number of
    players, each choosing from a pre-determined set of strategies, as well as the logic the defines the equilibria
    for the game.
    """
    DEFAULT_PARAMS = {}  #: the default parameters that are passed into the constructor by the L{GameDynamicsWrapper}
    PLAYER_LABELS = None  #: a list of labels to apply to each player in the game, used in graphing
    STRATEGY_LABELS = None  #: a list of lists of strings that name the available strategies for each player
    EQUILIBRIA_LABELS = ()  #: a list of labels corresponding to the integers returned by the classify function

    def __init__(self, payoff_matrices, player_frequencies, equilibrium_tolerance=0.1):
        """
        Initializes the game class with the give list of payoff matrices and distribution of players, as well as
        a notion of the equilibrium tolerance.

        @param payoff_matrices: a list of recursive lists representing the payoff matrix for each player, see L{PayoffMatrix} for more info
        @type payoff_matrices: list
        @param player_frequencies: a list that describes the distribution of players by player type, must sum to 1
        @type player_frequencies: list or tuple
        @param equilibrium_tolerance: the flexibility that should be used for equilibrium classification. An
            equilibrium is classified as such if 1 - equlibrium_tolerance proportion of people are playing a given
            set of strategies
        @type equilibrium_tolerance: float
        """
        assert payoff_matrices is not None
        assert player_frequencies is not None
        if self.PLAYER_LABELS is not None:
            assert len(player_frequencies) == len(self.PLAYER_LABELS)

        self.pm = PayoffMatrix(len(player_frequencies), payoff_matrices)
        if self.STRATEGY_LABELS is not None:
            for labels_i, num_strats  in zip(self.STRATEGY_LABELS, self.pm.num_strats):
                assert len(labels_i) == num_strats

        self.player_frequencies = player_frequencies
        self.equilibrium_tolerance = equilibrium_tolerance

    @classmethod
    def classify(cls, params, state, tolerance):
        """
        A class method that should be override by subclasses to help classify equilibria as a function of the current
        state of the population, the parameters to the game instance's constructor, and the equilibrium tolerance.

        @param params: An object that encapsulates all the parameters passed in to this object's constructor.
        @type params: L{Obj}
        @param state: a list of lists representing the distribution of players in each state
        @type state: list(list())
        @param tolerance: the equilibrium tolerance with which the instance was constructed
        @type tolerance: float
        @return: an integer representing the index of the equilibrium to which the state corresponds.
        @rtype: int
        """
        return -1

    @classmethod
    def num_equilibria(cls):
        """
        Get the number of equilibria for the game. This is one more than the number defined by the user.
        @return: the # of equilibria
        @rtype: int
        """
        return len(cls.EQUILIBRIA_LABELS) + 1

    @classmethod
    def get_equilibria(cls):
        """
        Get the list of equilibria defined by the class, plus the string representing the unclassified equlibrium, which
        can be easily accessed by indexing -1 on the tuple.

        @return: a tuple of strings of the equilibrium labels
        @rtype: tuple
        """
        return tuple(cls.EQUILIBRIA_LABELS) + (UNCLASSIFIED_EQUILIBRIUM, )


# common case is n =2, but we support as big N as needed
class SymmetricNPlayerGame(Game):
    """
    A convenience class that provides the logic for an N player game where each player chooses the from the same strategy
    set.
    """
    def __init__(self, payoff_matrix, n):
        """
        Initialize the symmetric game with the given payoff matrix and number of playeres

        @param payoff_matrix: a recursive list representing the payoff matrix for each player, see L{PayoffMatrix}
        @type payoff_matrix: list(list())
        @param n: the number of players in the game
        @type n: int
        """
        if self.STRATEGY_LABELS is not None:
            self.STRATEGY_LABELS = (self.STRATEGY_LABELS, ) * n

        # TODO: append as many as specified by num_players! not just one more
        # TODO: just need to think how to "transpose" a multidimensional matrix

        # interpreted as multiple instances of the same player, append the transpose
        payoff_matrix_2 = tuple(map(tuple, zip(*payoff_matrix))) # transpose
        matrices = [payoff_matrix, payoff_matrix_2]
        player_dist = (0.5, ) * n
        super(SymmetricNPlayerGame, self).__init__(payoff_matrices=matrices, player_frequencies=player_dist)



