__author__ = 'elubin'

from game import SymmetricNPlayerGame


class HawkDove(SymmetricNPlayerGame):
    DEFAULT_PARAMS = dict(v=30, c=60)
    STRATEGY_LABELS = ('Hawk', 'Dove')
    PLAYER_LABELS = ('Player 1', 'Player 2')

    def __init__(self, v, c):
        payoff_matrix = (((v - c) / 2.0, v),
                         (0, v / 2.0))

        super(HawkDove, self).__init__(payoff_matrix, 2)