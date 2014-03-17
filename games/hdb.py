__author__ = 'elubin'


from game import Game


class HawkDoveBourgeois(Game):
    DEFAULT_PARAMS = dict(v=30, c=60)

    def __init__(self, v, c):
        payoff_matrix_p1 = (((v - c) / 2.0, v, 3.0 * v / 4.0 - c / 4.0),
                            (0, v / 2.0, v / 4.0),
                            ((v - c) / 4.0, 3.0 * v / 4.0, v / 2.0))

        payoff_matrix_p2 = tuple(map(tuple, zip(*payoff_matrix_p1))) # transpose

        payoff_matrix = [payoff_matrix_p1, payoff_matrix_p2]
        player_dist = (0.5, 0.5)
        super(HawkDoveBourgeois, self).__init__(payoff_matrices=payoff_matrix, player_frequencies=player_dist)
