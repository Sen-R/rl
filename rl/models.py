from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def LinearReLUBlock(input_size, output_size, hidden_sizes, bn_pos = None,
                    output_transform = None, weight_init = None,
                    bias_init = None):
    """
    Returns a Module consisting of alternating fully connected and relu
    layers, with a custom output transformation and optional batch
    normalisation.

    Params
    ======
    input_size (int): size of input
    output_size (int): size of output layer
    hidden_sizes (list of ints): sizes of hidden layers
    bn_pos (list of ints): (optional) layer positions to insert batch
                           normalisation (position 0 corresponds to straight
                           after the first hidden layer, etc)
    output_transform (nn.Module): (optional) transformation to apply to the
                                  output
    weight_init (callable): (optional) initialiser to apply to all fully
                            connected layer's weight tensors.
    bias_init (callable): (optional) initialiser to apply fully connected
                          layers' bias tensors.
    """
    if bn_pos is None:
        bn_pos = []
        
    def initialized_linear_layer(i, o):
        layer = nn.Linear(i, o)
        if weight_init is not None:
            weight_init(layer.weight.data)
        if bias_init is not None:
            bias_init(layer.bias.data)
        return layer
    
    hidden_sizes = [input_size] + hidden_sizes
    layers = []
    for pos, (i, o) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
        layers.append(('fc_'+str(pos), initialized_linear_layer(i, o)))
        layers.append(('relu_'+str(pos), nn.ReLU(o)))
        if pos in bn_pos:
            layers.append(('bn_'+str(pos), nn.BatchNorm1d(o)))
    layers.append(('fc_out',
                   initialized_linear_layer(hidden_sizes[-1], output_size)))
    if output_transform is not None:
        layers.append(('out_trans', output_transform))
    return nn.Sequential(OrderedDict(layers))

class FCActor(nn.Module):
    """
    A fully connected actor network consisting of a "body" and "head"
    that are both LinearReLUBlocks, separated by a Batch Normalisation layer.
    """
    def __init__(self, state_size, action_size, body_hidden_sizes=[128],
                 head_hidden_sizes=[128]):
        super().__init__()
        self.body = LinearReLUBlock(state_size, body_hidden_sizes[-1],
                                    body_hidden_sizes[:-1])
        self.bn = nn.BatchNorm1d(body_hidden_sizes[-1])
        self.head = LinearReLUBlock(body_hidden_sizes[-1], action_size,
                                    head_hidden_sizes,
                                    output_transform = nn.Tanh())

    def forward(self, states):
        x = self.body(states)
        x = self.bn(x)
        actions = self.head(x)
        return actions

class FCCritic(nn.Module):
    """
    A fully connected critic network consisting of a "body" that only
    processes states, and a "head" that processes the intermediate state
    representation plus a candidate action, to deliver a value estimate.
    There is a Batch Normalisation layer applied to the outputs of the body.
    """
    def __init__(self, state_size, action_size, body_hidden_sizes=[128],
                 head_hidden_sizes=[128]):
        super().__init__()
        weight_init = None#lambda t: nn.init.uniform_(t, -1e-4, 1e-4)
        bias_init = None#nn.init.zeros_
        self.body = LinearReLUBlock(state_size, body_hidden_sizes[-1],
                                    body_hidden_sizes[:-1],
                                    weight_init=weight_init,
                                    bias_init=bias_init)
        self.bn = nn.BatchNorm1d(body_hidden_sizes[-1])
        self.head = LinearReLUBlock(body_hidden_sizes[-1]+action_size, 1,
                                    head_hidden_sizes,
                                    weight_init=weight_init,
                                    bias_init=bias_init)

    def forward(self, states, actions):
        x = self.body(states)
        x = self.bn(x)
        y = torch.cat((x, actions), dim=1)
        Q = self.head(y)
        return Q
