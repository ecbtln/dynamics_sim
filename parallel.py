
from joblib import Parallel, delayed


PARALLEL_ENABLED = True


def par_for(p=True):
    if p and PARALLEL_ENABLED:
        n_jobs = -1
    else:
        n_jobs = 1

    return Parallel(n_jobs=n_jobs)


def wrapper_simulate(wrapper, num_gens):
    return wrapper.simulate(num_gens=num_gens, graph=False, return_labeled=False)

def wrapper_vary_for_kwargs(wrapper, ips, idx, dependent_params, sim_wrapper, chosen_vals, **kwargs):
    return wrapper._vary_for_kwargs(ips, idx, dependent_params, sim_wrapper, chosen_vals, **kwargs)




