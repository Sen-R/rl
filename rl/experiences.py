from collections import deque
from collections.abc import Iterable
import numpy as np
import random

class TD0Experience:
    """ Class to encapsulate a single-step experience, e.g. for use in
    Q-learning """
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def to_dict(self):
        return {'state': self.state,
                'action': self.action,
                'reward': self.reward,
                'next_state': self.next_state,
                'done': self.done}
        
    def __repr__(self):
        return repr(self.to_dict())

    def __eq__(self, other):
        return all((np.array_equal(self.state, other.state),
                    np.array_equal(self.action, other.action),
                    np.array_equal(self.reward, other.reward),
                    np.array_equal(self.next_state, other.next_state),
                    np.array_equal(self.done, other.done)))

    @staticmethod
    def to_stacked(experiences):
        """ Static method to convert an iterable containing Experience objects
        into a tuple of vstacked states, actions rewards, next_states and
        dones """
        
        states = np.stack([e.state for e in experiences])
        actions = np.stack([e.action for e in experiences])
        rewards = np.stack([e.reward for e in experiences])
        next_states = np.stack([e.next_state for e in experiences])
        dones = np.stack([e.done for e in experiences]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    @staticmethod
    def from_stacked(states, actions, rewards, next_states, dones):
        """ Convert arrays into sequence of TD0Experiences """
        return (TD0Experience(s, a, r, ns, d) for s, a, r, ns, d in
                zip(states, actions, rewards, next_states, dones))

class RotatingList:
    """
    Rotating list -- maybe faster than deque for random access
    """
    def __init__(self, maxlen):
        self.storage = maxlen * [None]
        self.maxlen = maxlen
        self.cursor = 0
        self.len = 0

    def __len__(self):
        return self.len

    def __repr__(self):
        return repr(self.storage[:self.len])

    def __getitem__(self, idx):
        return self.storage[idx]#[:self.len][idx]

    def append(self, el):
        self.storage[self.cursor] = el
        self.len = min(self.maxlen, self.len+1)
        self.cursor = (self.cursor + 1) % self.maxlen

    def extend(self, seq):
        for el in seq:
            self.append(el)
        
    
class ExperienceReplayBuffer:
    """
    Basic experience replay buffer. TODO: add prioritisation.
    """
    
    def __init__(self, buffer_size):
        """
        buffer_size (int): maximum buffer size
        """
        self.buffer = deque(maxlen=buffer_size)

    @property
    def maxlen(self):
        return self.buffer.maxlen

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return repr(self.buffer)

    def record(self, experiences):
        """
        Add experience(s) to the buffer. Can work with batches or single
        experiences.
        """
        if isinstance(experiences, Iterable):
            self.buffer.extend(experiences)
        else:
            self.buffer.append(experiences)
            
    def replay(self, sample_size):
        """
        Return a random sample of experiences, of length `sample_size`.
        """
        ii = random.sample(range(len(self.buffer)), sample_size)
        return [self.buffer[i] for i in ii]
        
