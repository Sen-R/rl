from abc import ABC, abstractmethod
import numpy as np

class StochasticProcess(ABC):
    """ ABC for stochastic process generators """
    def __init__(self, x_init, random_state):
        self.rs = np.random.RandomState(random_state)
        self.x = np.copy(x_init)   
    
    @abstractmethod
    def sample(self):
        """
        Draw next sample
        """
        pass

class OUProcess(StochasticProcess):
    """ Generate samples from an OU process"""

    def __init__(self, x_inf, time_const, std_dev,
                 x_init=None, random_state=None):
        """
        Params
        ======
        x_inf (float or ndarray): Value to mean revert to. Also determines
                                  dimensions of noise (multi-dimensional
                                  process always uncorrelated)
        time_const (float): Mean reversion time constant, i.e. 1/theta,
                            determines length of auto-correlation of process
        std_dev (float): Long-term process standard deviation,
                         i.e. std_dev = sigma / sqrt(2*theta)
        x_init (float or ndarray): (optional) current value of process.
                                   Defaults to x_inf.
        random_state (None, int, array_like, RandomState): (optional)
                                                           random state
        """
        if x_init is None:
            x_init = x_inf
        super().__init__(x_init, random_state)
        self.x_inf = x_inf
        self.theta = 1 / time_const
        self.sigma = std_dev * np.sqrt(2*self.theta)

    def sample(self):
        """
        Draw next sample
        """
        dw = self.rs.normal(size=self.x.shape)
        dx = - self.theta * (self.x - self.x_inf) + self.sigma * dw
        self.x += dx
        return np.copy(self.x)
