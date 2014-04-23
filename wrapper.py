__author__ = 'elubin'

from plot import plot_data_for_players, plot_single_data_set, GraphOptions
from results import SingleSimulationOutcome
import inspect
import numpy


DEFAULT_ITERATIONS = 100
DEFAULT_GENERATIONS = 300

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

    def update_game_kwargs(self, *args, **kwargs):
        self.game_kwargs.update(*args, **kwargs)

    def update_dynamics_kwargs(self, *args, **kwargs):
        self.dynamics_kwargs.update(*args, **kwargs)

    def simulate(self, num_gens=100, graph=True, return_labeled=True):
        game = self.game_cls(**self.game_kwargs)
        dyn = self.dynamics_cls(payoff_matrix=game.pm,
                                player_frequencies=game.player_frequencies,
                                **self.dynamics_kwargs)
        results = dyn.simulate(num_gens=num_gens)
        #results_obj = SingleSimulationOutcome(self.dynamics_cls, self.dynamics_kwargs, self.game_cls, self.game_kwargs, results)
        # TODO: serialize results to file
        params = Obj(**self.game_kwargs)
        frequencies = numpy.zeros(self.game_cls.num_equilibria())  # one extra for the Unclassified key
        if dyn.stochastic:
            classifications = []
            for state in zip(*results):
                state = [x / x.sum() for x in state]
                equi = game.classify(params, state, game.equilibrium_tolerance)
                # note, if equi returns -1, then the -1 index gets the last entry in the array
                classifications.append(equi)
                frequencies[equi] += 1
        else:
            last_generation_state = results[-1]
            classification = game.classify(params, last_generation_state, game.equilibrium_tolerance)
            frequencies[classification] = 1

        if graph:
            graph_options = {}
            if game.STRATEGY_LABELS is not None:
                graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda p, s: game.STRATEGY_LABELS[p][s]

            if game.PLAYER_LABELS is not None:
                graph_options[GraphOptions.TITLE_KEY] = lambda p: game.PLAYER_LABELS[p]


            plot_data_for_players(results, range(num_gens), "Generation #", dyn.pm.num_strats,
                                  num_players=dyn.num_players,
                                  graph_options=graph_options)
        else:
            if return_labeled:
                return self._convert_equilibria_frequencies(frequencies)
            else:
                return frequencies

    # TODO: have another parameter, parallelize=False
    def simulate_many(self, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, return_labeled=True):
        frequencies = numpy.zeros(self.game_cls.num_equilibria())
        for iteration in range(num_iterations):
            results = self.simulate(num_gens=num_gens, graph=False, return_labeled=False)
            frequencies += results

        frequencies /= frequencies.sum()
        if return_labeled:
            return self._convert_equilibria_frequencies(frequencies)
        else:
            return frequencies

    @staticmethod
    def _static_convert_equilibria_frequencies(game_cls, frequencies):
        labels = game_cls.get_equilibria()
        return {label: freq for label, freq in zip(labels, frequencies) if freq > 0}

    def _convert_equilibria_frequencies(self, frequencies):
        return self._static_convert_equilibria_frequencies(self.game_cls, frequencies)



class IndependentParameter(object):
    def __init__(self, lb, ub, num_steps):
        """
        Construct an independent parameter. A varied simulation can have one or more independent parameters. Each
        independent parameter has a lower bound, an upper bound, the number of steps, a unique key identifying / labelling it
        (usually its param name for the game), a boolean indicating whether or not it is a direct input to the constructor,
        and a boolean indicating whether the parameter is for the dynamics constructor or game constructor.

        @param lb: The lower bound of the variation
        @type lb: int or float
        @param ub: The upper bound of the variation
        @type ub: int or float
        @param num_steps: the number of steps in between the lower bound and upper bound
        @type num_steps: int
        """
        self.lb = float(lb)
        self.ub = float(ub)
        self.num_steps = num_steps

    def __iter__(self):
        step = (self.ub - self.lb) / self.num_steps
        cur = self.lb
        while cur <= self.ub:
            yield cur
            cur += step


class DependentParameter(object):
    def __init__(self, func):
        """
        Each dependent parameter can be a function of both the values of all the other parameters, as well as,
        any other inputs. Some examples of when it would be easier to have a function of other inputs, not just
        the parameters is the following example.

        Thus, the passed in function should take three arguments.
            1. A dictionary of keyword arguments for the game constructor
            2. A dictionary of keyword arguments for the dynamics constructor
            3. A dictionary of indirect varied inputs, which may have been created during the varied game
        % Default values of temptation to defect in other simulations are:
        >>> default_cl= 4;
            default_ch= 12;

            % Generate vectors of temptations to defect that maintain mean but change
            % variance
            mean_c = p*default_cl + (1-p)*default_ch;
            cl_lower_bound = 3;
            cl_upper_bound = 12;
            ch_lower_bound = (mean_c - p*cl_upper_bound)/(1-p);
            ch_upper_bound = (mean_c - p*cl_lower_bound)/(1-p);

            cl_range = linspace(cl_upper_bound,cl_lower_bound,num_increments);
            ch_range = linspace(ch_lower_bound,ch_upper_bound,num_increments);

        In this case
        """
        self.func = func

    def get_val(self, **kwargs):
        """
        Evaluate the dependent parameter as a function of the other parameters for the namespace
        """
        return self.func(Obj(**kwargs))


class VariedGame(object):
    def __init__(self, game_cls, simulation_cls, game_kwargs=None, simulation_kwargs=None):
        self.game_cls = game_cls
        self.game_kwargs = game_kwargs if game_kwargs is not None else {}
        self.dynamics_cls = simulation_cls
        self.dynamics_kwargs = simulation_kwargs if simulation_kwargs is not None else {}

    def vary_param(self, kw, low, high, num_steps, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, graph=True):
        results = self.vary(game_kwargs={kw: (low, high, num_steps)}, num_iterations=num_iterations, num_gens=num_gens, graph=graph)[0]

        return results

    def vary(self, game_kwargs=None, dynamics_kwargs=None, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, graph=False):
        """
        We can vary the game kwargs, the dynamics kwargs, as well as any number of indirect inputs, if needed
        Each of these parameters must be an iterable of dictionaries, in the following form:

        game_kwargs = [{INDEPENDENT},{DEPENDENT}, {INDIRECT}]
        INDEPENDENT:
        Each key must be the string of the param name, as seen in the constructor
        Each value is an iterable of 3 values (lower_bound, upper_bound, num_steps)

        DEPENDENT:
        Each key must be the string of the param name, as seen in the constructor, cannot have any of the keys in the
        keys of the INDEPENDENT dict
        Each value is a function that takes in kwargs for the namespace

        INDIRECT:
        Each key must be the string of the param name, as seen in the constructor, cannot have any of the keys in the
        keys of the INDEPENDENT or DEPENDENT dicts


        If the root item is actually a dictionary, and not a list/tuple, then there are assumed to be no dependent kwargs or INDIRECT
        """
        assert not (game_kwargs is None and dynamics_kwargs is None), "nothing to vary!"
        if game_kwargs is None:
            game_kwargs = [{}, {}, {}]
        else:
            if isinstance(game_kwargs, dict):
                game_kwargs = [game_kwargs, {}, {}]
            else:
                assert isinstance(game_kwargs, (list, tuple))
                if len(game_kwargs) == 2:
                    game_kwargs.append({})
                assert len(game_kwargs) == 3
                # verify no duplicate keys
                key_set = set()

                for d in game_kwargs:
                    for k in d:
                        assert k not in key_set
                        key_set.add(k)



        if dynamics_kwargs is None:
            dynamics_kwargs = [{}, {}, {}]
        else:
            if isinstance(game_kwargs, dict):
                dynamics_kwargs = [dynamics_kwargs, {}, {}]
            else:
                assert isinstance(dynamics_kwargs, (list, tuple))
                if len(dynamics_kwargs) == 2:
                    game_kwargs.append({})
                assert len(dynamics_kwargs) == 3
                # verify no duplicate keys
                key_set = set()

                for d in dynamics_kwargs:
                    for k in d:
                        assert k not in key_set
                        key_set.add(k)


        assert len(game_kwargs[0]) > 0 or len(dynamics_kwargs[0]) > 0 or len(dynamics_kwargs[2]) > 0 or len(game_kwargs[2]) > 0, "We don't actually have any parameters to iterate over"

        local_game_kwarg_copy = self.game_kwargs.copy()
        game_kwargs_indirect = {}
        local_dynamic_kwarg_copy = self.dynamics_kwargs.copy()
        dynamic_kwargs_indirect = {}


        for kwargs in (game_kwargs, dynamics_kwargs):
            for i in (0, 2):
                for k in kwargs[i]:
                    v = kwargs[i][k]
                    assert len(v) == 3
                    kwargs[i][k] = IndependentParameter(*v)
            assert isinstance(kwargs[1], dict)
            for k in kwargs[1]:
                v = kwargs[1][k]
                argspec = inspect.getargspec(v)
                assert len(argspec.args) == 1
                kwargs[1][k] = DependentParameter(v)


        ips = []
        for outer_idx\
                , (kwargs, running_kwargs, running_indirect_kwargs) in enumerate(zip((game_kwargs, dynamics_kwargs),
                                                                                     (local_game_kwarg_copy, local_dynamic_kwarg_copy),
                                                                                     (game_kwargs_indirect, dynamic_kwargs_indirect))):
            for i in (0, 2):
                for k in kwargs[i]:
                    ip = (k, (running_kwargs, running_indirect_kwargs, kwargs[1], kwargs[i][k], i == 2, outer_idx))
                    ips.append(ip)

        w = GameDynamicsWrapper(self.game_cls, self.dynamics_cls, self.game_kwargs, self.dynamics_kwargs)
        results = []
        twod_data = []
        def perform_sim(game_kwargs, dynamics_kwargs):
            w.update_dynamics_kwargs(local_dynamic_kwarg_copy)
            w.update_game_kwargs(local_game_kwarg_copy)
            r = w.simulate_many(num_iterations=num_iterations, num_gens=num_gens, return_labeled=False)
            g_key = []
            for k in sorted(game_kwargs):
                g_key.append((k, game_kwargs[k]))

            d_key = []
            for k in sorted(dynamics_kwargs):
                d_key.append((k, dynamics_kwargs[k]))
            k = (tuple(g_key), tuple(d_key))
            # if k not in results:
            #     results[k] = []
            results.append(r)

            if len(ips) == 1:
                if len(twod_data) == 0:

                    twod_data.append(numpy.zeros((len(list(ips[0][1][3])), self.game_cls.num_equilibria())))
                    twod_data.append(0)
                twod_data[0][twod_data[1], :] = r
                twod_data[1] += 1

        self._vary_for_kwargs(ips, 0, perform_sim)

        if graph and len(ips) <= 2:
            graph_options = {}
            # if game.STRATEGY_LABELS is not None:
            #     graph_options[GraphOptions.STRATEGY_LABELS_KEY] = lambda p, s: game.STRATEGY_LABELS[p][s]
            #
            # if game.PLAYER_LABELS is not None:
            #     graph_options[GraphOptions.TITLE_KEY] = lambda p: game.PLAYER_LABELS[p]


            # we can only graph up to 1 or 2 varied parameters, so ignore the graph option if there's more
            if len(ips) == 1:
                graph_options[GraphOptions.LEGEND_LABELS_KEY] = lambda i: self.game_cls.get_equilibria()[i]
                plot_single_data_set(twod_data[0], ips[0][0], list(ips[0][1][3]), "Equilibrium Proportion", "Varying values of " + ips[0][0],  self.game_cls.num_equilibria(), graph_options)
            elif len(ips) == 2:
                #TODO: 3d graph, not as simple still doable
                pass
        return results


    def _vary_for_kwargs(self, ips, idx, perform_sim, varied_vals=None):
        """
        Each entry in remaining is a list of (key, (active_kwargs, indirect_kwargs, dict(dependent_parameters), IndependentParameter instance, indirect=true/false, dynamics=true/false))
        """
        if varied_vals is None:
            varied_vals = [{}, {}]

        if idx == len(ips):
            perform_sim(*varied_vals)
            return

        key, next = ips[idx]

        active_kwargs, indirect_kwargs, dependent_parameters, ip, indirect, is_dynamics = next

        for v in ip:
            varied_vals[is_dynamics][key] = v
            if indirect:
                indirect_kwargs[key] = v
            else:
                active_kwargs[key] = v

            # now recalculate all of the dependent parameters, as a function of both independent and indirect params

            for k in dependent_parameters:
                dp = dependent_parameters[k]
                kw = indirect_kwargs.copy()
                kw.update(active_kwargs)
                # only pass in the values of independent and indirect params, dependent params can't depend on other
                # dependent ones
                for temp_key in dependent_parameters:
                    if temp_key in kw:
                        del kw[temp_key]

                if k in kw:
                    del kw[k]

                active_kwargs[k] = dp.get_val(**kw)

            self._vary_for_kwargs(ips, idx + 1, perform_sim, varied_vals)




