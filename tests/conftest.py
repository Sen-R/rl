import pytest

import torch

@pytest.fixture(scope="module")
def d_states():
    return torch.tensor([1, 3, 6, 2])

@pytest.fixture(scope="module")
def d_state():
    return torch.tensor(3)

@pytest.fixture(scope="module")
def d_state_size():
    return 7

@pytest.fixture(scope="module")
def d_action_size():
    return 4

@pytest.fixture(scope="module")
def prefs():
    return torch.tensor([[0., 0., 0., 0.]] * 6 + [[1., 2., -1., -3.]])

@pytest.fixture(scope="module")
def state_action_probs():
    return torch.tensor([[0.25, 0.25, 0.25, 0.25]] * 2 +
                        [[0.2583, 0.7020, 0.0350, 0.0047]] +
                        [[0.25, 0.25, 0.25, 0.25]])
