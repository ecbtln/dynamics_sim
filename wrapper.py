__author__ = 'elubin'

from plot import plot_data_for_players, GraphOptions
from results import SingleSimulationOutcome


class Obj:
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])


class GameDynamicsWrapper(object):
    def __init__(self, game_cls, dynamics_cls, game_kwargs=None, dynamics_kwargs=None):
        self.game_kwargs = game_cls.DEFAULT_PARAMS
        if game_kwargs is not None:
            self.game_kwargs.update(game_kwargs)

        if dynamics_kwargs is None:
            dynamics_kwargs = {}
        self.game_cls = game_cls
        self.dynamics_cls = dynamics_cls
        self.dynamics_kwargs = dynamics_kwargs

    def set_game_kwargs(self, **kwargs):
        self.game_kwargs.update(kwargs)

    def set_dynamics_kwargs(self, **kwargs):
        self.dynamics_kwargs.update(kwargs)

    def simulate(self, num_gens=100, graph=True):
        game = self.game_cls(**self.game_kwargs)
        dyn = self.dynamics_cls(payoff_matrix=game.pm,
                                player_frequencies=game.player_frequencies,
                                **self.dynamics_kwargs)
        results = dyn.simulate(num_gens=num_gens)
        #results_obj = SingleSimulationOutcome(self.dynamics_cls, self.dynamics_kwargs, self.game_cls, self.game_kwargs, results)
        # TODO: serialize results to file
        params = Obj(**self.game_kwargs)
        frequencies = {}
        if dyn.stochastic:
            classifications = []
            for state in zip(*results):
                state = [x / x.sum() for x in state]
                equi = game.classify(params, state, game.equilibrium_tolerance)
                classifications.append(equi)
                frequencies[equi] = frequencies.get(equi, 0) + 1
        else:
            last_generation_state = results[-1]
            classification = game.classify(params, last_generation_state, game.equilibrium_tolerance)
            frequencies[classification] = 1

        if graph:
            graph_options = {}
            if game.STRATEGY_LABELS is not None:
                graph_options[GraphOptions.STRATEGY_LABELS_KEY] = lambda p, s: game.STRATEGY_LABELS[p][s]

            if game.PLAYER_LABELS is not None:
                graph_options[GraphOptions.TITLE_KEY] = lambda p: game.PLAYER_LABELS[p]


            plot_data_for_players(results, range(num_gens), "Generation #", dyn.pm.num_strats,
                                  num_players=dyn.num_players,
                                  graph_options=graph_options)
        else:
            return frequencies

    # TODO: have another parameter, parallelize=False
    def simulate_many(self, num_iterations=1000, num_gens=100):
        frequencies = {}
        for iteration in range(num_iterations):
            results = self.simulate(num_gens=num_gens, graph=False)
            for k in results:
                frequencies[k] = frequencies.get(k, 0) + results[k]

        s = 0
        for k in frequencies:
            s += frequencies[k]

        for k in frequencies:
            frequencies[k] /= float(s)

        return frequencies



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
