__author__ = 'elubin'


class PayoffMatrix(object):
    """
    A class that encapsulates the notion of a set of payoff matrices for a game (one for each player), and provides
    convenience methods for getting the payoff for each player given a strategy set, as well as calculating the
    expected payoff given a distribution of players playing each strategy.
    """
    def __init__(self, num_players, payoff_matrices):
        self.num_player_types = num_players

        self.payoff_matrices = payoff_matrices
        self.num_strats = []
        root = self.payoff_matrices[0]
        for i in range(self.num_player_types):
            self.num_strats.append(len(root))
            root = root[0]
        self.verify_payoff_matrix_dimensions()

    def verify_payoff_matrix_dimensions(self):
        """
        Verify that "depth" of each payoff matrix matches number of elements in player_dist
        """
        for m in self.payoff_matrices:
            self._verify_dimensions(m, self.num_strats[:])

    def _verify_dimensions(self, m, num_strats):
        """
        Recursive helper function to verify the dimensions of the payoff matrix
        """
        if len(num_strats) == 0:
            assert isinstance(m, (int, float))
            return
        n = num_strats.pop(0)
        assert n == len(m)
        for i in m:
            self._verify_dimensions(i, num_strats[:])

    def get_payoff(self, recipient, *strats):
        """
        Get the payoff for the player index recipient, by specifying the strategies that everyone plays in increasing
        player order.

        @param recipient: the index of the player for which to get the payoff, 0-indexed
        @type recipient: int
        @param strats: the iterable of strategies played by each player, in the order of their indices
        @type strats: list(int)
        @return: the payoff that the recipient gets from all playres playing the given strategy
        @rtype: float
        """
        matrix = self.payoff_matrices[recipient]
        for idx in strats:
            matrix = matrix[idx]
        return matrix

    def get_expected_payoff(self, player_idx, strategy, current_state):
        """
        Get the expected payoff if the player at idx player_idx plays indexed by strategy given the current state.
        @param player_idx: the index of the player for which to get the expected payoff
        @type player_idx: int
        @param strategy: the index that the player will play
        @type strategy: int
        @param current_state: The state of the population(s). Each entry in the parent array refers to a player type, each entry in each sublist refers to the number or
            frequency of players playing that strategy.
        @type current_state: list
        @return: the expected payoff
        @rtype: float
        """
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
                p = float(n) / current_state[current_player_idx].sum()
                dict_copy = other_player_stategies.copy()
                #dict_copy = dict(other_player_stategies.items())
                dict_copy[current_player_idx] = strat
                payoff += self._iterate_through_players(target_player_idx, current_player_idx + 1, dict_copy, probability * p, current_state)

            return payoff
