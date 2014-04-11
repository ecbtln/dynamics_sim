__author__ = 'elubin'


# a class that helps handling the array of payoff matrices for each player in a game

class PayoffMatrix(object):
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
        # verify that "depth" of each payoff matrix matches number of elements in player_dist
        for m in self.payoff_matrices:
            self._verify_dimensions(m, self.num_strats[:])

    def _verify_dimensions(self, m, num_strats):
        if len(num_strats) == 0:
            assert isinstance(m, (int, float))
            return
        n = num_strats.pop(0)
        assert n == len(m)
        for i in m:
            self._verify_dimensions(i, num_strats[:])

    def get_payoff(self, recipient, *strats):
        """
        Get the payoff for the player index recipient, by specifiying the strategies that everyone plays in increasing
        player order

        """
        matrix = self.payoff_matrices[recipient]
        for idx in strats:
            matrix = matrix[idx]
        return matrix

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
                p = float(n) / current_state[current_player_idx].sum()
                dict_copy = dict(other_player_stategies.items())
                dict_copy[current_player_idx] = strat
                payoff += self._iterate_through_players(target_player_idx, current_player_idx + 1, dict_copy, probability * p, current_state)

            return payoff
