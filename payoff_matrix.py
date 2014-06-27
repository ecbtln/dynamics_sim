__author__ = 'elubin'
import numpy
import logging


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
        self.compute_dominated_strategies()

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

    def get_all_strategy_tuples(self):
        """
        @return: a generator of all strategy tuples representing non-mixed strategies for all players
        @rtype: generator
        """
        return self._strategy_tuple_helper(0, ())

    def _strategy_tuple_helper(self, p, s):
        if p == self.num_player_types:
            yield s
            return

        for s_i in range(self.num_strats[p]):
            for r in self._strategy_tuple_helper(p + 1, s + (s_i, )):
                yield r

    def compute_dominated_strategies(self):
        # for every strategy for every player, iterate through all strategies for all other players and see if there are
        # any strategies that are completely dominated by other strategies
        # dominated is a dictionary of sets

        # we have a loop to simulate the iterated elimination of dominated strategies
        continue_iterating = True
        dominated_strategies = set()

        while continue_iterating:
            continue_iterating = False
            for p_i in range(self.num_player_types):
                payoffs = []
                for s_i in range(self.num_strats[p_i]):
                    payoffs.append(numpy.array(self._get_all_payoffs(p_i, s_i, dominated_strategies)))


                for s_1 in range(self.num_strats[p_i]):
                    if (p_i, s_1) in dominated_strategies:
                        continue
                    # consider s_1 as a dominated strategy
                    for s_2 in range(self.num_strats[p_i]):
                        # if s_2 is dominated, we can ignore it. can't both be be dominated and dominate another one
                        if (p_i, s_2) in dominated_strategies:
                            continue

                        if (payoffs[s_2] > payoffs[s_1]).all():
                            dominated_strategies.add((p_i, s_1))
                            continue_iterating = True
                            break
        # (player index, strategy) index set of tuples of dominated strategies
        self.dominated_strategies = set()




    def _get_all_payoffs(self, p, s, dominated):
        # get a list of all possible payoffs a given player can get for playing a given strategy
        # the list is computed in order, by iterating through all the other strategy pairs for all the other players
        # TODO need to ignore payoffs for dominated strategies
        return list(self._get_all_payoffs_helper(p, s, 0, (), dominated))

    def _get_all_payoffs_helper(self, p, s, cur_p, cur_s, dominated):
        if cur_p == self.num_player_types:
            yield self.get_payoff(p, *cur_s)
            return
        elif cur_p == p:
            for r in self._get_all_payoffs_helper(p, s, cur_p + 1, cur_s + (s, ), dominated):
                yield r
            return
        else:
            for s_i in range(self.num_strats[cur_p]):
                if (cur_p, s_i) in dominated:
                    continue
                for r in self._get_all_payoffs_helper(p, s, cur_p + 1, cur_s + (s_i, ), dominated):
                    yield r

    def is_pure_equilibrium(self, s):

        assert self.num_player_types == len(s)
        strategies = list(s)
        for n_i in range(self.num_player_types):

            best_payoff = self.get_payoff(n_i, *s)
            for s_i in range(self.num_strats[n_i]):
                if s_i == s[n_i]:
                    continue
                strategies[n_i] = s_i
                p = self.get_payoff(n_i, *strategies)
                if p > best_payoff:
                    return False, n_i, s_i  # profitable deviation for player n_i to play s_i instead of s[n_i]

            strategies[n_i] = s[n_i]

        return True

    def is_mixed_equilibrium(self, s):
        assert self.num_player_types == len(s)
        logging.debug("testing %s", s)
        for n_i in range(self.num_player_types):
            logging.debug("player %d", n_i)
            payoffs = []
            for i, s_i in enumerate(s[n_i]):
                if s_i > 0:
                    # get expected payoff of mixing this strategy
                    payoffs.append((i, self.get_expected_payoff(n_i, i, s)))
            logging.debug("payoffs %s", payoffs)
            if len(payoffs) > 1:
                for i, (idx_i, p) in enumerate(payoffs):
                    for j, (idx_j, q) in enumerate(payoffs[i:]):
                        if abs(q - p) > (1e-08 + 1e-05 * abs(p)):
                            return True, n_i, ((idx_i, p), (idx_j, q))

            else:
                # only one strategy, this is pure equilibrium
                # check to make sure there's no incentive of switching strategies

                # get index of pure strategy
                s_idx = [i for i, x in enumerate(s[n_i]) if s[n_i][i] > 0][0]

                best_payoff = self.get_expected_payoff(n_i, s_idx, s)
                logging.debug("Best payoff %f", best_payoff)
                for s_i in range(self.num_strats[n_i]):
                    if s_i == s_idx:
                        continue
                    p = self.get_expected_payoff(n_i, s_i, s)
                    logging.debug("Strategy %d payoff %f", s_i, p)
                    if p > best_payoff:
                        return False, n_i, s_i  # profitable deviation for player n_i to play s_i instead of s[n_i]

        return True