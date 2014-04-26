__author__ = 'elubin'

from plot import plot_data_for_players, plot_single_data_set, GraphOptions
from results import NDimensionalData
import inspect
import numpy
import types, marshal
from parallel import par_for, delayed, wrapper_simulate, wrapper_vary_for_kwargs


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

            graph_options[GraphOptions.NO_MARKERS_KEY] = True

            plot_data_for_players(results, range(num_gens), "Generation #", dyn.pm.num_strats,
                                  num_players=dyn.num_players,
                                  graph_options=graph_options)
        else:
            if return_labeled:
                return self._convert_equilibria_frequencies(frequencies)
            else:
                return frequencies

    def simulate_many(self, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, return_labeled=True, parallelize=True):
        frequencies = numpy.zeros(self.game_cls.num_equilibria())
        output = par_for(parallelize)(delayed(wrapper_simulate)(self, num_gens=num_gens) for iteration in range(num_iterations))

        for x in output:
            frequencies += x

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

    def _step_size(self):
        return (self.ub - self.lb) / self.num_steps

    def __getitem__(self, item):
        if item >= 0 and item <= self.num_steps:
            return self.lb + item * self._step_size()
        elif item < 0 and item >= -self.num_steps - 1:
            return self.ub + (1 + item) * self._step_size()
        else:
            raise IndexError

    def __len__(self):
        return self.num_steps + 1


class VerboseIndependentParameter(IndependentParameter):
    """
    An extension on the IndependentParameter class that makes room for three other properties
        - The key that the independent parameter is varying
        - Whether or not the parameter is direct or indirect (indirect may be used as params to dependent params, but don't directly get applied to the class constructor)
        - Whether the parameter is for the dynamics or the class constructor
    """
    def __init__(self, key, is_direct, is_game_kwarg, *args, **kwargs):
        self.key = key
        self.is_direct = is_direct
        self.is_game_kwarg = is_game_kwarg
        super(VerboseIndependentParameter, self).__init__(*args, **kwargs)


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
        assert func.func_closure is None, "In order to support parallelization, the lambda must NOT be a closure. It can only be a function of the parameters to the simulation."
        self.func = func

    def get_val(self, **kwargs):
        """
        Evaluate the dependent parameter as a function of the other parameters for the namespace
        """
        return self.func(Obj(**kwargs))

    # Hack to allow portability of lambdas cross-process.
    # The only requirement is that the function doesn't have a closure, which we check above
    def __getstate__(self):
        return {'func': marshal.dumps(self.func.func_code)}

    def __setstate__(self, state):
        self.func = types.FunctionType(marshal.loads(state['func']), globals())


class VariedGame(object):
    def __init__(self, game_cls, simulation_cls, game_kwargs=None, simulation_kwargs=None):
        self.game_cls = game_cls
        self.game_kwargs = game_kwargs if game_kwargs is not None else {}
        self.dynamics_cls = simulation_cls
        self.dynamics_kwargs = simulation_kwargs if simulation_kwargs is not None else {}

    def vary_param(self, kw, (low, high, num_steps), **kwargs):
        return self.vary(game_kwargs={kw: (low, high, num_steps)}, graph=True, **kwargs)

    def vary_2params(self, kw1, (low1, high1, num_steps1), kw2, (low2, high2, num_steps2), **kwargs):
        return self.vary(game_kwargs={kw1: (low1, high1, num_steps1), kw2: (low2, high2, num_steps2)}, graph=True, **kwargs)

    def vary(self, game_kwargs=None, dynamics_kwargs=None, num_iterations=DEFAULT_ITERATIONS, num_gens=DEFAULT_GENERATIONS, graph=False, parallelize=True):
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

        kwargs = [game_kwargs, dynamics_kwargs]

        for j, kw in enumerate(kwargs):
            if kw is None:
                kwargs[j] = [{}, {}, {}]
            else:
                if isinstance(kw, dict):
                    kwargs[j] = [kw, {}, {}]
                else:
                    assert isinstance(kw, (list, tuple))
                    if len(kw) == 2:
                        kw.append({})
                    assert len(kw) == 3
                    # verify no duplicate keys
                    key_set = set()

                    for d in kw:
                        for k in d:
                            assert k not in key_set
                            key_set.add(k)

        assert len(kwargs[0][0]) > 0 or len(kwargs[1][0]) > 0 or len(kwargs[1][2]) > 0 or len(kwargs[0][2]) > 0, "We don't actually have any parameters to iterate over"

        independent_params = []
        for i, kw in enumerate(kwargs):
            for j in (0, 2):
                for k in kw[j]:
                    v = kw[j][k]
                    assert len(v) == 3
                    ip = VerboseIndependentParameter(k, j == 0, i == 0, *v)
                    kw[j][k] = ip
                    independent_params.append(ip)
            assert isinstance(kw[1], dict)
            for k in kw[1]:
                v = kw[1][k]
                argspec = inspect.getargspec(v)
                assert len(argspec.args) == 1
                kw[1][k] = DependentParameter(v)

        w = GameDynamicsWrapper(self.game_cls, self.dynamics_cls, self.game_kwargs, self.dynamics_kwargs)

        dependent_params = (kwargs[0][1], kwargs[1][1])
        results = self._vary_kwargs(independent_params, dependent_params, w, num_iterations=num_iterations, num_gens=num_gens, parallelize=parallelize)

        data = NDimensionalData.initialize(results, independent_params)

        # TODO: persist results
        if graph:
            data.graph(self.game_cls.get_equilibria())

        return results

    def _vary_kwargs(self, ips, dependent_params, sim_wrapper, **kwargs):
        return self._vary_for_kwargs(ips, 0, dependent_params, sim_wrapper, (), **kwargs)

    def _vary_for_kwargs(self, ips, idx, dependent_params, sim_wrapper, chosen_vals, parallelize=False, **kwargs):
        """
        A recursively defined function to iterate over all possible permutations of the variables defined in the list
        of independent variables that returns the simulation results of the cross product of these variable variations.

        @param ips: a list of all the VerboseIndependentParameters that will be varied
        @type ips: list(VerboseIndependentParameter)
        @param idx: the index of the independent parameter about to be iterated upon
        @type idx: int
        @param dependent_params: the tuple of dictionaries representing the DependentParameters for the game_kwargs
            and the dynamics_kwargs, respectively.
        @type dependent_params: tuple({string: DependentParameter})
        @param sim_wrapper: the pre-initialized sim-wrapper on which we will call simulate_many
        @type sim_wrapper: GameDynamicsWrapper
        @param chosen_vals: a tuple of all the indices of the chosen values for each already-decided independent param
        @type chosen_vals: tuple(int)
        @param parallelize: whether or not to parallelize the subloops of this function. We set to true on the parent call
            and then false for all recursive calls.
        @type parallelize: bool
        @param kwargs: These are the rest of the keyword arguments that should be passed directly to the simulate_many function call
        @rtype: list(list(...))
        @return: a recursive list of lists representing the simulation results for having assigned each independent parameter
            the value corresponding to the index at which the simulation results are present in the list of lists.

            i.e. Two independent parameters, the return type will be a list of lists of simulation results

            The simulation result present at the address [4][17] represents the value of the simulation when the first
            independent parameter was set to its value at index 4 (@see IndependentParameter.__getitem__), and the second
            independent parameter was set to its value at index 17.
        """

        if idx == len(ips):
            # the list is divided as follows:
            # [[direct_game_kwargs, indirect_game_kwargs], [direct_dynamics_kwargs, indirect_dynamics_kwargs]]
            varied_kwargs = [[{}, {}], [{}, {}]]

            # helper function to return the correct keywords give the desired params
            def kws(ip):
                return varied_kwargs[int(not ip.is_direct)][int(not ip.is_game_kwarg)]

            for chosen_idx, ip in zip(chosen_vals, ips):
                kws(ip)[ip.key] = ip[chosen_idx]

            # the list is organized as follows:
            # [game_kwargs, dynamics_kwargs]
            sim_kwargs = [{}, {}]

            for i in (0, 1):
                # set all the direct ones
                for k, v in varied_kwargs[i][0].items():
                    sim_kwargs[i][k] = v

                # now calculate all of the dependent parameters, as a function of both direct
                # and indirect independent parameters
                for k, dp in dependent_params[i].items():
                    # get the inputs to the dependent param calculation
                    if i == 0:
                        dependent_kw_params = sim_wrapper.game_kwargs.copy()
                    else:
                        dependent_kw_params = sim_wrapper.dynamics_kwargs.copy()
                    dependent_kw_params.update(varied_kwargs[i][0])
                    dependent_kw_params.update(varied_kwargs[i][1])

                    sim_kwargs[i][k] = dp.get_val(**dependent_kw_params)

            sim_wrapper.update_dynamics_kwargs(sim_kwargs[1])
            sim_wrapper.update_game_kwargs(sim_kwargs[0])
            # don't paralellize the simulate_many requests, we are parallelizing higher up in the call chain
            return sim_wrapper.simulate_many(return_labeled=False, parallelize=False, **kwargs)

        var_indices = xrange(len(ips[idx]))
        #dependent_params = [{}, {}]
        return par_for(parallelize)(delayed(wrapper_vary_for_kwargs)(self, ips, idx + 1, dependent_params, sim_wrapper, chosen_vals + (i, ), **kwargs) for i in var_indices)




