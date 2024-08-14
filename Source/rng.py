# This file will be used to define the random number generator class

from abc import ABC, abstractmethod
import numpy as np

class RNG(ABC):
    @abstractmethod
    def get_integers(self, low, high):
        pass

    @abstractmethod
    def get_normal(self, loc, scale):
        pass

    @abstractmethod
    def get_random(self):
        pass

    @abstractmethod
    def get_seed(self, seed):
        pass

    @abstractmethod
    def get_uniform(self, low, high):
        pass

    @abstractmethod
    def check_probablity(self, p):
        pass

    @abstractmethod
    def get_binomial(self, n, p):
        pass


class NumpyRNG(RNG):
    def __init__(self, seed=None):
        self.seed(seed)

    def get_integers(self, low, high):
        return np.random.randint(low, high)

    def get_normal(self, loc, scale):
        return np.random.normal(loc, scale) # loc = mean, scale = std

    def get_random(self):
        return np.random.random()

    def get_seed(self, seed):
        np.random.seed(seed)

    def get_uniform(self, low, high):
        return np.random.uniform(low, high)

    def check_probablity(self, p):
        return np.random.random() < p

    def get_binomial(self, n, p):
        return np.random.binomial(n, p)
    

    