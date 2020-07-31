import pytest

from rl.agents import *
from rl.models import *

class TestRandomAgent:
    def test_functional(self):
        agent = RandomAgent(4)
        action = agent.act(np.array([1., 2.]))
        assert action.shape == (4,)
        actions = agent.act(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        assert actions.shape == (3, 4)

class TestDDPGAgent:
    def test_functional(self, tmp_path):
        agent = DDPGAgent(2, 4, 1, FCActor, FCCritic, batch_size=2,
                          update_after=2)
        action = agent.act(np.array([1., 2.]))
        assert action.shape == (4,)
        actions = agent.act(np.array([[1., 2.], [3., 4.]]))
        assert actions.shape == (2, 4)
        s = np.array(4*[[1., 2.]])
        a = np.array(4*[[1., 2., 3., 4.]])
        r = 4*[1.2]
        ns = np.array(4*[[3., 4.]])
        d = 4*[1.]
        assert not agent.time_to_learn
        agent.observe(s, a, r, ns, d)
        assert len(agent.buffer)==4
        print(agent.update_after)
        assert agent.time_to_learn
        agent.save(str(tmp_path / "ddpg.pt"))
        agent_l = DDPGAgent(2, 4, 1, FCActor, FCCritic, batch_size=2,
                          update_after=2)
        agent_l.load(str(tmp_path / "ddpg.pt"))
        for s, l in zip(agent.policy.parameters(), agent_l.policy.parameters()):
            assert torch.allclose(s, l)
        for s, l in zip(agent.Q.parameters(), agent_l.Q.parameters()):
            assert torch.allclose(s, l)
        

    def load_save(self):
        agent_s = DDPGAgent(2, 4, 1, FCActor, FCCritic, batch_size=2,
                            update_after=2)
        agent_l = DDPGAgent(2, 4, 1, FCActor, FCCritic, batch_size=2,
                            update_after=2)
        
