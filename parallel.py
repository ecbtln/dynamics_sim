
from joblib import Parallel, delayed


PARALLEL_ENABLED = True #: Whether or not parallelization should be enabled for the library


def par_for(p=True):
    """
    A convenience function that uses the job-lib library to create a wrapper for a sub-process paralleization job.
    Depending on whether or not p is true or false, a wrapper is created to parallelize the loop on all available cores,
    or to simply run it in serial.

    @param p: whether or not to parallelize
    @type p: bool
    """
    if p and PARALLEL_ENABLED:
        n_jobs = -1
    else:
        n_jobs = 1

    return Parallel(n_jobs=n_jobs)


def wrapper_simulate(wrapper, num_gens):
    """
    The multiprocessing library requires that the function called is globally importable, and it doesn't work with
    instance methods. Therefore, this is he method corresponding to the L{GameDynamicsWrapper} classes's simulate method
    so that multiple simulations can be run in parallel. This function does not need to be called directly.
    """
    return wrapper.simulate(num_gens=num_gens, graph=False, return_labeled=False)


def wrapper_vary_for_kwargs(wrapper, *args, **kwargs):
    """
    The multiprocessing library requires that the function called is globally importable, and it doesn't work with
    instance methods. Therefore, this is he method corresponding to the L{VariedGame} classes's vary_for_kwargs method
    so that multiple simulations can be run in parallel. This function does not need to be called directly.
    """
    return wrapper._vary_for_kwargs(*args, **kwargs)







