import pytest
from rl.policies import *

### DATA ###

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

    def test_pref_shape(self, d_state_size, d_action_size):
        p = TabularPolicy(d_state_size, d_action_size)
        assert list(p.prefs.shape) == [d_state_size, d_action_size]
        
    def test_call_v(self, d_state_size, d_action_size, d_states):
        p = TabularPolicy(d_state_size, d_action_size)
        probs = p(d_states)
        assert list(probs.shape) == [len(d_states), d_action_size]
        assert torch.allclose(probs, torch.tensor(1./d_action_size))

    def test_call_s(self, d_state_size, d_action_size, d_state):
        p = TabularPolicy(d_state_size, d_action_size)
        probs = p(d_state)
        assert list(probs.shape) == [d_action_size]
        assert torch.allclose(probs, torch.tensor(1./d_action_size))
        
    def test_call2(self, d_state_size, d_action_size, d_states, prefs,
                   state_action_probs):
        p = TabularPolicy(d_state_size, d_action_size)
        p.prefs.data.copy_(prefs)
        probs = p(d_states)
        assert torch.allclose(probs, state_action_probs, rtol=0.01, atol=0.001)

    def test_stoch_act(self, d_state_size, d_action_size, d_states, prefs,
                       state_action_probs):
        p = TabularPolicy(d_state_size, d_action_size)
        p.prefs.data.copy_(prefs)
        actions = p.stoch_act(d_states)
        assert isinstance(actions, torch.distributions.Categorical)
        assert torch.allclose(actions.probs, state_action_probs,
                              rtol=0.01, atol=0.001)

    def test_act(self, d_state_size, d_action_size, d_states, d_state):
        p = TabularPolicy(d_state_size, d_action_size)
        actions = p.act(d_states)
        assert list(actions.shape) == [len(d_states)]
        action = p.act(d_state)
        assert list(action.shape) == []

    def test_parameters(self, d_state_size, d_action_size):
        p = TabularPolicy(d_state_size, d_action_size)
        params = {k: v for k, v in p.named_parameters()}
        assert params == {'prefs': p.prefs}
            
        

