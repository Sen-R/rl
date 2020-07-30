import pytest
from rl.experiences import *

class TestTD0Experience:
    def test_construction(self):
        s, a, r, ns, d = 1, 3, 0.1, 2, True
        e = TD0Experience(s, a, r, ns, d)
        assert e.state == s
        assert e.action == a
        assert e.reward == r
        assert e.next_state == ns
        assert e.done == d

    def test_to_dict(self):
        s, a, r, ns, d = 1, 3, 0.1, 2, True
        ed = TD0Experience(s, a, r, ns, d).to_dict()
        assert ed['state'] == s
        assert ed['action'] == a
        assert ed['reward'] == r
        assert ed['next_state'] == ns
        assert ed['done'] == d

    def test_repr(self):
        e = TD0Experience(1, 3, 0.1, 2, True)
        assert repr(e)==repr(e.to_dict())

    def test_stacked(self):
        states = np.tensordot([1, 2, 3, 4], np.ones((2,2)), axes=0)
        actions = np.array([2, 4, 6, 1])
        rewards = np.array([5, 6, 7, 8])
        next_states = np.roll(states, axis=0, shift=1)
        dones = np.array([0, 0, 1, 0])
        print("original:\n=========")
        print(states, actions, rewards, next_states, dones, sep='\n\n')
        print("\nexperiences:\n============")
        experiences = list(TD0Experience.from_stacked(states, actions, rewards,
                                                 next_states, dones))
        print(experiences)
        expected = [TD0Experience(*e) for e in zip(states, actions, rewards,
                                                   next_states, dones)]
        print(expected)
        for act, exp in zip(experiences, expected):
            assert act == exp
        nps, npa, npr, npns, npd = TD0Experience.to_stacked(experiences)
        print("\n\nstacked:\n========")
        print(nps, npa, npr, npns, npd, sep='\n\n')
        for e, v in zip([states, actions, rewards, next_states, dones],
                        [nps, npa, npr, npns, npd]):
            assert v.shape == e.shape
            assert np.array_equal(v, e)

class TestRotatingList:
    def test_basics(self):
        l = RotatingList(2)
        assert len(l.storage)==2
        assert l.maxlen == 2
        assert l.cursor == 0
        assert l.len == 0
        l.append('a')
        assert repr(l)==repr(['a'])
        l.append('b')
        assert l.cursor == 0
        assert l.len == 2
        assert l.maxlen == 2
        assert l.storage == ['a', 'b']
        l.append('c')
        assert l.cursor == 1
        assert l.len == 2
        assert l.maxlen == 2
        assert l.storage == ['c', 'b']
        assert l[1] == 'b'

    def test_extend(self):
        l = RotatingList(3)
        l.extend(['a', 'b', 'c', 'd'])
        assert l.storage == ['d', 'b', 'c']
            
class TestExperienceReplayBuffer:
    def test_basics(self):
        erb = ExperienceReplayBuffer(3)
        assert erb.maxlen==3
        assert len(erb)==0
        assert repr(erb)==repr(erb.buffer)

    def test_record(self):
        single = 0
        multiple = [1, 2, 3]
        erb = ExperienceReplayBuffer(3)
        erb.record(single)
        assert erb.buffer[-1] == single
        erb.record(multiple)
        assert list(erb.buffer) == multiple

    def test_replay(self):
        data = [1., 2., 3., 4., 5.]
        erb = ExperienceReplayBuffer(3)
        erb.record(data)
        sample = erb.replay(2)
        assert len(sample)==2
        assert len(set(sample) - set(data))==0
        
