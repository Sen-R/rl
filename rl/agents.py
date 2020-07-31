from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from .stochasticprocesses import Scrambler, OUScrambler
from .policies import DeterministicModulePolicy
from .experiences import TD0Experience, ExperienceReplayBuffer
from .utils import soft_update, hard_update
from .estimators import sarsa_estimate
from .trainers import DeterministicPolicyTrainer, ValueFunctionTrainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(ABC):
    """ ABC for an agent. Defines minimal interface. """
    @abstractmethod
    def act(self, states):
        """
        Return action(s) for provided state(s). Should work on and return
        numpy arrays.
        """

    @abstractmethod
    def observe(self, states, actions, rewards, next_states, dones):
        """
        Agent observes consequences of taking `actions` in `states` -- i.e.
        what `rewards`, `next_states` and `dones` result.
        """

class RandomAgent(Agent):
    """
    Random agent, draws IID random actions from normal distribution,
    and clips them.
    """

    def __init__(self, action_size, mu=0., sigma=1., lb=-1., ub=1.,
                 random_state=None):
        self.action_size = action_size
        self.mu = mu
        self.sigma = sigma
        self.lb = lb
        self.ub = ub
        self.rs = np.random.RandomState(random_state)
        self.train_mode = False

    def act(self, states):
        action_shape = (self.action_size if len(states.shape)==1 else
                        (len(states), self.action_size))
        actions = self.rs.normal(self.mu, self.sigma, action_shape)
        return np.clip(actions, self.lb, self.ub)

    def observe(self, states, actions, rewards, next_states, dones):
        pass # Do nothing: no learned parameters

class LearningAgent(Agent):
    """
    ABC to define interface for any RL agent controlled by a torch-based
    Policy.
    """
    def __init__(self, policy, noise=None, train_mode=True):
        """
        `noise` generator, if used, must return samples of shape
        (action_size,) or (num_agents, action_size). NB noise only generated
        in `train_mode`.

        Params
        ======
        policy (Policy): policy
        noise (None or Scrambler): noise generator
        train_mode (bool): whether agent is set to learn
        """
        self.policy = policy
        self.noise = noise
        self.train_mode = train_mode
    
    def act(self, states):
        states = torch.from_numpy(states).float().to(device)
        squeeze = (len(states.shape)==1)
        if squeeze:
            states.unsqueeze_(0)
        self.policy.eval()
        with torch.no_grad():
            actions = self.policy.act(states).cpu().data.numpy()
        self.policy.train()
        if self.train_mode and self.noise is not None:
            actions = self.noise(actions)
        if squeeze:
            actions = actions.squeeze()
        return actions

    def observe(self, states, actions, rewards, next_states, dones):
        """ Observe outcomes of actions, record in memory whatever
        is required for learning and, if it is time, learn """

        self.record(states, actions, rewards, next_states, dones)
        
        if self.train_mode and self.time_to_learn:
            self.learn()

    @property
    @abstractmethod
    def time_to_learn(self):
        """
        Implement to determine which observations steps should be used
        as an opportunity to learn.
        """
            
    @abstractmethod
    def record(self, states, actions, rewards, next_states, dones):
        """Implement whatever agent does to record its observations"""
            
    @abstractmethod
    def learn(self):
        """Implement whatever agent does to learn from previous experiences."""

    @abstractmethod
    def load(self, data):
        """Implement to load saved parameters."""

    @abstractmethod
    def save(self):
        """Implement to save learned parameters."""
        
    
class DDPGAgent(LearningAgent):
    """ DDPG agent """

    def __init__(self, state_size, action_size, num_agents,
                 actor, critic, actor_kw=None, critic_kw=None,
                 buffer_size=100000, batch_size=128, update_after=1000,
                 gamma=0.99, tau=0.001, lr_actor=3e-4, lr_critic=3e-4,
                 l2_actor=1e-4, l2_critic=1e-4,
                 noise_std_dev=0.4, noise_time_const=10,
                 train_mode=True):
        """
        Initialise DDPG agent. Continuous valued actions are assumed to
        take values between -1 and 1.

        Params
        ======
        state_size (int): dimension of state space
        action_size (int): dimension of action space
        num_agents (int): number of agents to control
        actor (fn): constructor for actor network, assumed to take state_size
        critic (fn): constructor for critic network, assumed to take state_size
                     and action_size as first two positional arguments
        actor_kw (dict): (optional) keyword arguments for actor constructor
        critic_kw (dict): (optional) keyword arguments for critic constructor
        buffer_size (int): size of experience replay buffer
        batch_size (int): number of experiences to use in single learning step
        update_after (int): number of experiences to collect before starting
                            learning; could be overridden to be at least the
                            batch_size
        gamma (float): discount factor for future rewards
        tau (float): soft_update parameter for updating target networks
        lr_actor (float): learning rate for actor network
        lr_target (float): learning rate for target network
        l2_actor (float): L2 penalty for actor network weights
        l2_critic (float): L2 penalty for critic network weights
        noise_std_dev (float or callable): std dev of OU noise to add to action.
            If callable, assumed to be a function of the number of steps.
        noise_time_const (float): time const for OU noise to add to action
        train_mode (bool): learn from experiences
        """
        if actor_kw is None:
            actor_kw = {}
        if critic_kw is None:
            critic_kw = {}

        policy = DeterministicModulePolicy(actor(state_size, action_size,
                                                 **actor_kw).to(device))
        self.policy_target = DeterministicModulePolicy(actor(state_size,
                                                             action_size,
                                                             **actor_kw).to(device))
        self.Q = critic(state_size, action_size, **critic_kw).to(device)
        self.Q_target = critic(state_size, action_size, **critic_kw).to(device)
        hard_update(policy, self.policy_target)
        hard_update(self.Q, self.Q_target)

        noise = OUScrambler(num_agents, action_size, noise_time_const,
                            noise_std_dev)
        super().__init__(policy, noise, train_mode)

        self.buffer = ExperienceReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.update_after = max(batch_size, update_after)
        self.gamma = gamma
        self.tau = tau

        policy_optimizer = optim.Adam(self.policy.parameters(),
                                      lr = lr_actor, weight_decay = l2_actor)

        self.policy_trainer = DeterministicPolicyTrainer(self.policy,
                                                         self.Q,
                                                         policy_optimizer)

        Q_optimizer = optim.Adam(self.Q.parameters(),
                                 lr = lr_critic, weight_decay = l2_critic)
        self.Q_trainer = ValueFunctionTrainer(self.Q,
                                              Q_optimizer) #TODO check

    @property
    def time_to_learn(self):
        return len(self.buffer) >= self.update_after

    def record(self, states, actions, rewards, next_states, dones):
        experiences = TD0Experience.from_stacked(states, actions, rewards,
                                                 next_states, dones)
        self.buffer.record(experiences)

    def learn(self):
        # Extract sample experiences from memory
        batch = self.buffer.replay(self.batch_size)
        exp_tuple = tuple(torch.from_numpy(c).float().to(device)
                          for c in TD0Experience.to_stacked(batch))
        states, actions, rewards, next_states, dones = exp_tuple

        # Estimate corresponding action values
        with torch.no_grad():
            Q_targets = sarsa_estimate(rewards, next_states, dones,
                                       self.Q_target, self.policy_target,
                                       discrete_actions=False)

        # Train Q and policy networks
        self.Q_trainer.train(Q_targets, states, actions)
        soft_update(self.Q, self.Q_target, self.tau)
        self.policy_trainer.train(states)
        soft_update(self.policy, self.policy_target, self.tau)
        
    def save(self, filename):
        """
        Save agent's learned parameters to filename.
        """
        torch.save({'policy': self.policy.state_dict(),
                    'Q': self.Q.state_dict()},
                   filename)

    def load(self, filename):
        """
        Load agent's learned parameters from filename. NB assumes you are
        loading weights to an agent with exactly the same policy and Q
        networks as for the agent you saved.
        """
        params = torch.load(filename)
        self.policy.load_state_dict(params['policy'])
        self.Q.load_state_dict(params['Q'])

