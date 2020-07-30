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

class GaussianWhiteNoiseProcess(StochasticProcess):
    """ Generate Gaussian white noise samples """
    def __init__(self, mu, sigma, random_state=None):
        """
        Params
        ======
        mu (float or array): process mean
        sigma (float or array): process std_dev
        random_state (None, int, array_like, RandomState): (optional) random
                                                           state
        """
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        super().__init__(x_init=mu, random_state=random_state)

    def sample(self):
        """Draw next sample"""
        self.x = self.rs.normal(self.mu, self.sigma)
        return np.copy(self.x)

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
        #self.theta = 1 / time_const
        #self.sigma = std_dev * np.sqrt(2*self.theta)
        self.time_const = time_const
        self.std_dev = std_dev

    @property
    def time_const(self):
        return 1. / self.theta

    @time_const.setter
    def time_const(self, val):
        self.theta = 1. / val

    @property
    def std_dev(self):
        return self.sigma / np.sqrt(2. * self.theta)

    @std_dev.setter
    def std_dev(self, val):
        self.sigma = val * np.sqrt(2. * self.theta)

    def sample(self):
        """
        Draw next sample
        """
        dw = self.rs.normal(size=self.x.shape)
        dx = - self.theta * (self.x - self.x_inf) + self.sigma * dw
        self.x += dx
        return np.copy(self.x)

class Scrambler(ABC):
    """ ABC for classes that scramble actions by adding random noise """

    @abstractmethod
    def __call__(self, actions):
        """ Implement to scramble actions """

class AdditiveNoiseScrambler(Scrambler):
    """
    Class that adds a stochastic process to (continuous-valued) action
    vectors and then clips output between `lb` and `ub`.
    """
    def __init__(self, process, lb=-1., ub=1.):
        self.process = process
        self.lb = lb
        self.ub = ub

    def __call__(self, actions):
        actions += self.process.sample()
        actions = actions.clip(self.lb, self.ub)
        return actions

    def _required_shape(self, num_agents, action_size):
        return (num_agents, action_size)
        
class OUScrambler(AdditiveNoiseScrambler):
    def __init__(self, num_agents, action_size, time_const, std_dev, lb=-1.,
                 ub=1., random_state=None):
        x_inf = np.zeros(self._required_shape(num_agents, action_size))
        process = OUProcess(x_inf, time_const, std_dev,
                            random_state=random_state)
        super().__init__(process, lb, ub)

class GaussianWhiteNoiseScrambler(AdditiveNoiseScrambler):
    def __init__(self, num_agents, action_size, std_dev, lb=-1., ub=1.,
                 random_state=None):
        shape = self._required_shape(num_agents, action_size)
        mu = np.zeros(shape)
        sigma = std_dev * np.ones(shape)
        process = GaussianWhiteNoiseProcess(mu, sigma, random_state)
        super().__init__(process, lb, ub)
