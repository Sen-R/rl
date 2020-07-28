import pytest
import numpy as np
from rl.policies import *

d_states = torch.tensor([1, 3, 6, 2])
d_state = torch.tensor(4)

prefs = torch.tensor([[0., 0., 0., 0.]] * 6 + [[1., 2., -1., -3.]])

state_action_probs =  torch.tensor([[0.25, 0.25, 0.25, 0.25]] * 2 +
                                   [[0.2583, 0.7020, 0.0350, 0.0047]] +
                                   [[0.25, 0.25, 0.25, 0.25]])

state_size = 7
action_size = 4

class TestPolicy:
    def test_construction_fails(self):
        with pytest.raises(TypeError):
            p = Policy()

class TestStochasticPolicy:
    def test_construction_fails(self):
        with pytest.raises(TypeError):
            p = StochasticPolicy()

class TestTabularPolicy:
    def test_construction(self):
        p = TabularPolicy(3, 2)

    def test_pref_shape(self):
        p = TabularPolicy(state_size, action_size)
        assert list(p.prefs.shape) == [state_size, action_size]

    def test_call(self):
        p = TabularPolicy(state_size, action_size)
        probs = p(d_states)
        assert list(probs.shape) == [len(d_states), action_size]
        assert torch.allclose(probs, torch.tensor(1./action_size))
        probs = p(d_state)
        assert list(probs.shape) == [action_size]
        assert torch.allclose(probs, torch.tensor(1./action_size))
        p.prefs.data.copy_(prefs)
        probs = p(d_states)
        assert torch.allclose(probs, state_action_probs, rtol=0.01, atol=0.001)
        
    def test_stoch_act(self):
        p = TabularPolicy(state_size, action_size)
        p.prefs.data.copy_(prefs)
        actions = p.stoch_act(d_states)
        assert isinstance(actions, torch.distributions.Categorical)
        assert torch.allclose(actions.probs, state_action_probs,
                              rtol=0.01, atol=0.001)

    def test_act(self):
        p = TabularPolicy(state_size, action_size)
        actions = p.act(d_states)
        assert list(actions.shape) == [len(d_states)]
        action = p.act(d_state)
        assert list(action.shape) == []

    def test_parameters(self):
        p = TabularPolicy(state_size, action_size)
        params = {k: v for k, v in p.named_parameters()}
        assert params == {'prefs': p.prefs}
            
a = [0., 0., 1., 0.]

def q(states):
    squeeze = len(states.shape)==0
    n_states = 1 if squeeze else states.shape[0]
    qvectors = torch.tensor([a]).repeat(n_states, 1)
    if squeeze:
        qvectors.squeeze_()
    return qvectors    
        
class TestDiscretePolicies:
    def test_construction(self):
        pi = RandomDiscretePolicy(action_size, seed=None)
        pi = GreedyPolicy(q) 
        pi = EpsilonGreedyPolicy(len(a), q, 0., seed=24)

    @pytest.mark.parametrize("policy", [RandomDiscretePolicy(1, seed=42),
                                        EpsilonGreedyPolicy(len(a),
                                                            q, 0.,
                                                            seed=42)])
    def test_seed_setting(self, policy):
        expected = [0.88226926, 0.91500396, 0.38286376]
        generated = [torch.rand([], generator=policy.rs).numpy()
                     for _ in expected]
        assert np.allclose(expected, generated, atol=1e-8)

    @pytest.mark.parametrize("states,shape", [(3, []),
                                              ([3], [1]),
                                              ([4, 1], [2])])
    def test_act_shape(self, states, shape):
        pi = EpsilonGreedyPolicy(len(a), q, 0.)
        actions = pi.act(torch.tensor(states))
        assert list(actions.shape)==shape

    @pytest.mark.parametrize("pi", [RandomDiscretePolicy(1, seed=42),
                                    GreedyPolicy(q),
                                    EpsilonGreedyPolicy(len(a),
                                                        q, 0.,
                                                        seed=42)])
    def test_act_range(self, pi):
        states = range(20) # values don't matter
        actions = pi.act(torch.tensor(states))
        assert torch.all(actions >= 0)
        assert torch.all(actions < len(a))
        
    @pytest.mark.slow
    @pytest.mark.random
    @pytest.mark.parametrize("epsilon",
                             [0., 0.25, 0.5, 0.75, 1.])
    def test_epsilon_greedy(self, epsilon):
        n_inner = 10000
        n_outer = 10
        n_total = n_inner * n_outer
        best_action = np.argmax(a)
        best_selected = 0
        pi = EpsilonGreedyPolicy(action_size, q, epsilon,
                                 seed=int(278633*epsilon))
        for _ in range(n_outer):
            actions = pi.act(torch.zeros(n_inner))
            best_selected += torch.sum(actions==best_action).numpy()
        p_best = (1. - epsilon) + epsilon / len(a)
        expected = p_best * n_total
        tol = 1.96 * np.sqrt(p_best * (1.-p_best) * n_total) # 1 S.D.
        assert np.abs(best_selected - expected) <= tol

    def test_greedy(self):
        best_action = torch.tensor(np.argmax(a))
        best_selected = 0
        pi = GreedyPolicy(q)
        actions = pi.act(torch.zeros(5))
        assert torch.allclose(actions, best_action)

class TestModulePolicies:
    state_size = 2
    action_size = 2
    state = torch.tensor([1., 2.])
    states = torch.tensor([[1., 2.], [3., 4.]])
    act_module = nn.Sequential(nn.Linear(state_size, action_size))

    class MSModule(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.mean_layer = nn.Linear(state_size, action_size)
            self.std_layer = nn.Linear(state_size, action_size)
        def forward(self, states):
            return self.mean_layer(states), self.std_layer(states)

    ms_module = MSModule(state_size, action_size)

    disc_policy = DiscreteModulePolicy(act_module)
    det_policy = DeterministicModulePolicy(act_module)
    norm_policy = NormalModulePolicy(ms_module)

    def test_dists(self):
        assert isinstance(self.disc_policy.stoch_act(self.states),
                          distributions.Categorical)
        assert isinstance(self.norm_policy.stoch_act(self.states),
                          distributions.Normal)
    
    def test_actions(self):
        assert (list(self.disc_policy.act(self.state).shape) == [])
        assert (list(self.disc_policy.act(self.states).shape) ==
                [len(self.states)])
        assert (list(self.det_policy.act(self.states).shape) ==
                [len(self.states), self.action_size])
        assert (list(self.norm_policy.act(self.states).shape) ==
                [len(self.states), self.action_size])
        
        
