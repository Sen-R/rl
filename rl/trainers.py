from torch import nn

class ValueFunctionTrainer:
    """
    Class for training a value function (e.g. state value, action value,
    advantage, etc). This class just orchestrates training, given a value
    estimate, optimizer, loss function, etc.
    """
    def __init__(self, value_function, optimizer, loss_fn=None):
        """
        Params
        ======
        value_function (nn.Module): differentiable function approximator
        optimizer (optim.Optimizer): optimizer object
        loss_fn (None, Loss): loss criterion, default is nn.MSELoss
        """
        self.value_function = value_function
        self.optimizer = optimizer
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        self.loss_fn = loss_fn

    def train(self, targets, states, actions=None):
        """
        Params
        ======
        targets (Tensor): target values to train towards
        states (Tensor): corresponding states
        actions (None, Tensor): corresponding actions (if applicable)

        Returns
        =======
        losses (Tensor): losses (e.g. for debugging purposes)
        """
        if actions is None:
            predictions = self.value_function(states).squeeze()
        else:
            predictions = self.value_function(states, actions).squeeze()
        losses = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
class DeterministicPolicyTrainer:
    """
    Class for training differentiable deterministic policies to learn the
    optimal action for a given value function. Can be used to implement
    DDPG.

    You need to supply the policy, value network, optimizer etc. This class
    just orchestrates the training.
    """
    def __init__(self, policy, action_evaluator, optimizer):
        """
        Params
        ======
        policy (DeterministicModulePolicy): policy to train
        action_evaluator (nn.Module): action-value / action-advantage function
        optimizer (optim.Optimizer): optimizer object
        """
        self.policy = policy
        self.action_evaluator = action_evaluator
        self.optimizer = optimizer

    def train(self, states):
        """
        Params
        ======
        states (Tensor): states to use for this training step
        """
        actions = self.policy.act(states)
        neg_action_values = -self.action_evaluator(states, actions).mean()
        self.optimizer.zero_grad()
        neg_action_values.backward()
        self.optimizer.step()
