import warnings
import functools
import pickle
import pandas as pd
import time

class SeedShifter:
    def __init__(self, random_seed, number_of_loaders):
        self.random_seed = random_seed
        self.number_of_loaders = number_of_loaders

    def get_seed(self, phase, epoch_number):
        if phase in ("train", "training"):
            return self.random_seed + ((self.number_of_loaders + 1) * epoch_number)
        elif phase in ("val", "validation"):
            return self.random_seed + (10 ** 6) + ((self.number_of_loaders + 1) * epoch_number)
        elif phase in ("test",):
            return self.random_seed + (2 * (10 ** 6)) + ((self.number_of_loaders + 1) * epoch_number)
        else:
            raise KeyError(phase)

    @classmethod
    def from_parameters(cls, parameters):
        return cls(
            random_seed=parameters['random_seed'],
            number_of_loaders=parameters['number_of_loaders'],
        )

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def unpickle_from_file(file_name):
    with open(file_name, 'rb') as handle:
        try:
            return pickle.load(handle)
        except ImportError:
            return pd.read_pickle(file_name)

def get_random_seed(random_seed):
    if random_seed == -1:
        return int(time.time() * 1000000) % (2 ** 30)
    else:
        return random_seed
