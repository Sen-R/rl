from collections import deque
from collections.abc import Iterable
import numpy as np

class TDExperience:
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

    @staticmethod
    def stacked(experiences):
        """ Static method to convert an iterable containing Experience objects
        into a tuple of vstacked states, actions rewards, next_states and
        dones """
        
        states = np.stack([e.state for e in experiences])
        actions = np.stack([e.action for e in experiences])
        rewards = np.stack([e.reward for e in experiences])
        next_states = np.stack([e.next_state for e in experiences])
        dones = np.stack([e.done for e in experiences]).astype(np.uint8)

        return states, actions, rewards, next_states, dones


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
        return np.random.choice(self.buffer, sample_size)
        
