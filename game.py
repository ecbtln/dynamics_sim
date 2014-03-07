__author__ = 'elubin'
from plot import plot_data_for_players


class Game(object):
    def __init__(self, payoff_matrix, player_frequencies, dynamics=None, **dyn_init_kwargs):
        self.payoff_matrix = payoff_matrix
        self.player_frequencies = player_frequencies
        self.dynamics = dynamics
        self.init_kwargs = dyn_init_kwargs

    def set_dynamics(self, cls_name, **kwargs):
        self.dynamics = cls_name
        self.init_kwargs = kwargs

    def simulate_once(self, num_gens=100, graph_options=None):
        assert self.dynamics is not None
        dyn = self.dynamics(payoff_matrix=self.payoff_matrix,
                            player_frequencies=self.player_frequencies,
                            **self.init_kwargs)
        history = dyn.simulate(num_gens=num_gens)
        plot_data_for_players(history, range(num_gens), "Generation #", dyn.num_players, graph_options=graph_options)


    def simulate_many(self, num_sums=500):
        # TODO: process results
        pass



class VariedGame(object):
    def __init__(self, simulation_cls, simulation_kwargs, game_cls, game_kwargs):
        def new_game(**kwargs):
            kw = game_kwargs.copy()
            kw.update(kwargs)
            return game_cls(kw)

        self.game_constructor = new_game
        self.simulation_cls = simulation_cls
        self.simulation_kwargs = simulation_kwargs

    def vary_params(self, kw, kw_range, num_sims=500):
        for kw_v in kw_range:
            g = self.game_constructor(kw=kw_v)
            g.set_dynamics(self.simulation_cls, **self.simulation_kwargs)
            # TODO: process results
            results = g.simulate_many(num_sims=num_sims)
            

