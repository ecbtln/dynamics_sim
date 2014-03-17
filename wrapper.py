__author__ = 'elubin'

from plot import plot_data_for_players
from results import SingleSimulationOutcome


class GameDynamicsWrapper(object):
    def __init__(self, game_cls, dynamics_cls, game_kwargs=None, dynamics_kwargs=None):
        if game_kwargs is None:
            game_kwargs = game_cls.DEFAULT_PARAMS
        if dynamics_kwargs is None:
            dynamics_kwargs = {}
        self.game_cls = game_cls
        self.game_kwargs = game_kwargs
        self.dynamics_cls = dynamics_cls
        self.dynamics_kwargs = dynamics_kwargs

    def set_game_kwargs(self, **kwargs):
        self.game_kwargs.update(kwargs)

    def set_dynamics_kwargs(self, **kwargs):
        self.dynamics_kwargs.update(kwargs)

    def simulate(self, num_gens=100):
        game = self.game_cls(**self.game_kwargs)
        dyn = self.dynamics_cls(payoff_matrix=game.pm,
                                player_frequencies=game.player_frequencies,
                                **self.dynamics_kwargs)
        results = dyn.simulate(num_gens=num_gens)
        results_obj = SingleSimulationOutcome(self.dynamics_cls, self.dynamics_kwargs, self.game_cls, self.game_kwargs, results)
        # TODO: serialize results to file
        # if dyn.stochastic:
        #     classifications = []
        #     frequencies = {}
        #     for state in results:
        #         equi = game.classify(self.game_kwargs, state, game.equilibrium_tolerance)
        #         classifications.append(equi)
        #         frequencies[equi] = frequencies.get(equi, 0) + 1
        #     print frequencies
        # else:
        #     last_generation_state = results_obj.last_generation()
        #     classification = game.classify(self.game_kwargs, last_generation_state, game.equilibrium_tolerance)
        #     print classification

        # TODO graph results
        plot_data_for_players(results, range(num_gens), "Generation #", dyn.num_players)


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
        # TODO: should be able to vary parameters of dynamics as well, if you want
        for kw_v in kw_range:
            g = self.game_constructor(**{kw: kw_v})
            g.set_dynamics(self.simulation_cls, **self.simulation_kwargs)
            # TODO: process results
            results = g.simulate_many(num_sims=num_sims)
