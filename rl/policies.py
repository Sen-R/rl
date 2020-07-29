"""
This module contains abstract base classes containing the building blocks
from which we will build RL agents
"""

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F

class Policy(ABC):
    """ ABC to represent an agent's policy """

    @abstractmethod
    def act(self, states):
        """ Return action(s) given state(s) """
        pass


class StochasticPolicy(Policy):
    """ ABC to represent a policy that generate random actions. Provides
    an extra method to return the random action itself as a distribution
    object. """

    def __init__(self, pathwise):
        super().__init__()
        self.pathwise = pathwise

    def act(self, states):
        d = self.stoch_act(states)
        if not self.pathwise:
            actions = d.sample()
        else:
            if not d.has_rsample:
                raise RuntimeError('distribution does not support rsample')
            actions = d.rsample()
        return actions

    @abstractmethod
    def stoch_act(self, states):
        """ Implement in derived class. Should convert states into
        corresponding action distributions """
        pass
        
class TabularPolicy(nn.Module, StochasticPolicy):
    """
    Stochastic policy over discrete state and action spaces. Parameters
    are a 2-dimensional Tensor, each row representing the soft-max action
    preferences for a given state. The policy is also a torch Module,
    allowing easy implementation of policy gradient methods.
    """

    def __init__(self, state_size, action_size):
        nn.Module.__init__(self)
        StochasticPolicy.__init__(self, pathwise=False)
        self.prefs = nn.Parameter(torch.zeros((state_size, action_size),
                                              requires_grad=True))

    def forward(self, states):
        """ Return probability vector(s) """
        return F.softmax(self.prefs[states], dim=-1)

    def stoch_act(self, states):
        """ Return action(s) """
        return distributions.Categorical(self(states))

class RandomDiscretePolicy(Policy):
    """
    Random policy over discrete action space.

    NB although this is really a stochastic policy, we never differentiate
    through the stochasticity, so we subclass from Policy.

    Params
    ======
    action_size (int): size of action space
    seed (None, int): (optional) random seed
    """

    def __init__(self, action_size, seed=None):
        super().__init__()
        self.action_size = action_size
        self.rs = torch.Generator()
        if seed is not None:
            self.rs.manual_seed(seed)

    def act(self, states):
        actions_shape = () if len(states.shape)==0 else (states.shape[0],)
        return torch.randint(0, self.action_size, actions_shape,
                             generator=self.rs)

class GreedyPolicy(Policy):
    """
    Greedy policy over discrete action space.
    """
    
    def __init__(self, q):
        """
        Params
        ======
        q (nn.Module): module that maps state(s) to vector(s) of action-values
        """
        super().__init__()
        self.q = q

    def act(self, states):
        return torch.argmax(self.q(states), dim=-1)
        
class EpsilonGreedyPolicy(Policy):
    """
    Epsilon greedy policy over discrete action space.

    NB although this is really a stochastic policy, we will implement it
    as a deterministic policy + noise, as the policy has no learnable
    parameters so we'll never want to backpropagate through the policy."""

    def __init__(self, action_size, q, epsilon, seed=None):
        """
        Params
        ======
        action_size (int): size of action space
        q (nn.Module): module that maps state(s) to vector(s) of action-values
        epsilon (float): probability of taking random action
        seed (int): (optional) random seed
        """
        super().__init__()
        self.random_policy = RandomDiscretePolicy(action_size, seed)
        self.greedy_policy = GreedyPolicy(q)
        self.epsilon = epsilon
        self.rs = torch.Generator()
        if seed is not None:
            self.rs.manual_seed(seed)

    def act(self, states):
        best_actions = self.greedy_policy.act(states)
        random_actions = self.random_policy.act(states)
        be_greedy = torch.rand(best_actions.size(),
                               generator=self.rs) >= self.epsilon
        return torch.where(be_greedy, best_actions, random_actions)
    
class DiscreteModulePolicy(nn.Module, StochasticPolicy):
    """
    A stochastic policy for a discrete action space
    """
    def __init__(self, pi):
        """
        Params
        ======
        pi (nn.Module): Module that maps states to soft-max action preferences
        """
        nn.Module.__init__(self)
        StochasticPolicy.__init__(self, pathwise=False)
        self.pi = pi

    def stoch_act(self, states):
        probs = F.softmax(self.pi(states), dim=-1)
        return distributions.Categorical(probs)

class DeterministicModulePolicy(nn.Module, Policy):
    """
    Provide a Policy interface to a deterministic policy module. Output is
    an action vector or disc
    """
    def __init__(self, pi):
        """
        Params
        ======
        pi (nn.Module): Module that maps states to actions
        """
        nn.Module.__init__(self)
        Policy.__init__(self)
        self.pi = pi

    def act(self, states):
        return self.pi(states)

class NormalModulePolicy(nn.Module, StochasticPolicy):
    """
    A stochastic policy for a continuous action space. Output actions are
    independent and normally distributed.
    """
    def __init__(self, pi, pathwise=None):
        """
        Params
        ======
        pi (nn.Module): Module that maps states to a tuple of action means and
                        standard deviations
        pathwise (bool): If true, use `rsample` to draw differentiable samples
        """
        nn.Module.__init__(self)
        StochasticPolicy.__init__(self, pathwise=pathwise)
        self.pi = pi

    def stoch_act(self, states):
        means, stds = self.pi(states)
        stds = F.relu(stds) # Safety
        return distributions.Normal(means, stds)

class FixedActionVectorPolicy(Policy):
    """
    Policy that emits the same action vector regardless of state.
    """
    def __init__(self, action_vector):
        """
        Params
        ======
        action (array): fixed action vector to emit
        """
        self.action_vector = torch.Tensor(action_vector)

    def act(self, states):
        if len(states.shape)==1: # single state
            actions = self.action_vector
        else:
            actions = self.action_vector.repeat(states.shape[0], 1)
        return actions
    
