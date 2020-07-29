import pytest
from rl.estimators import *

q_d = torch.tensor([0., 0., 1., -1.])

def Q_discrete(states):
    if len(states.shape)==1:
        return q_d
    else:
        return q_d.repeat(states.shape[0], 1)

class ForcePolicy:
    def __init__(self, action):
        self.action = action
    def act(self, states):
        if len(states.shape)==1:
            return torch.tensor(self.action, dtype=torch.long)
        else:
            return self.action * torch.ones(states.shape[0],
                                            dtype=torch.long)
    
def Q_continuous(states, actions):
    squeeze = len(states.shape)==1
    if squeeze:
        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
    Q = torch.sum(states ** 2, dim=1) + torch.sum(actions ** 2, dim=1)
    if squeeze:
        Q.squeeze_()
    return Q

class ContinuousPolicy:
    def act(self, states):
        a = torch.tensor([0.35])
        if len(states.shape)==1:
            return a
        else:
            return a.repeat(states.shape[0], 1)

@pytest.mark.parametrize("rewards,next_states",
                         [(0.1, [1.]), ([0.1, -0.2], [[1, 2], [3, 4]])])
@pytest.mark.parametrize("next_action", [0, 1, 2, 3])
def test_sarsa_discrete(rewards, next_states, next_action):
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    gamma = 0.9
    exp_value = sarsa_estimate(rewards, next_states, Q_discrete,
                               ForcePolicy(next_action),
                               gamma, discrete_actions=True)
    next_actions = ForcePolicy(next_action).act(next_states)
    should_be = rewards + gamma * q_d[next_action]*torch.ones(len(next_states))
    assert torch.allclose(exp_value, should_be)

@pytest.mark.parametrize("rewards, next_states",
                         [([0.5], [[0.1, 0.3]]), (0.5, [0.1, 0.3]),
                          ([0.2, 0.1], [[0.1, 0.2], [0.3, 0.4]])])
def test_sarsa_continuous(rewards, next_states):
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    gamma = 0.9
    policy = ContinuousPolicy()
    exp_value = sarsa_estimate(rewards, next_states, Q_continuous,
                               policy, gamma, discrete_actions=False)
    should_be = rewards + gamma * Q_continuous(next_states,
                                               policy.act(next_states))
    assert torch.allclose(exp_value, should_be)
    
@pytest.mark.todo
def test_sarsamax():
    pass

@pytest.mark.todo
def test_expected_sarsa():
    pass

@pytest.mark.todo
def test_td0():
    pass
