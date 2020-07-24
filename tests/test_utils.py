import pytest
from rl.utils import *

import torch
from torch import nn

l_init = -2.
t_init = 2.

def local_nn():
    l = nn.Linear(3, 2)
    nn.init.constant_(l.weight, l_init)
    nn.init.constant_(l.bias, l_init)
    return l

def target_nn():
    t = nn.Linear(3, 2)
    nn.init.constant_(t.weight, t_init)
    nn.init.constant_(t.bias, t_init)
    return t

def test_hard_update():
    local = local_nn()
    target = target_nn()
    hard_update(local, target)
    for l, t in zip(local.parameters(), target.parameters()):
        assert torch.allclose(l, t)

@pytest.mark.parametrize("tau", [0., 0.25, 1.])
def test_soft_update(tau):
    local = local_nn()
    target = target_nn()
    print('Before')
    print('------')
    for l, t in zip(local.parameters(), target.parameters()):
        print (l, t, sep='\n\n')
    soft_update(local, target, tau)
    print('\nAfter')
    print(  '-----')
    for l, t in zip(local.parameters(), target.parameters()):
        e = torch.ones_like(t) * (tau * l_init + (1-tau) * t_init)
        print (l, t, e, sep='\n\n')
        assert torch.allclose(t, e)
    
    
