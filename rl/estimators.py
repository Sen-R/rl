import torch

def sarsa_estimate(rewards, next_states, Q, policy, gamma=1.,
                   discrete_actions=True):
    """
    Sarsa estimator of the value of the current state-action pair given
    observations of the reward and next state.

    Can work on single observations or batches.

    Note this function expects the `policy` to be consistent with Q (i.e. it
    won't check this). True Sarsa requires `policy` to be the policy currently
    used by the agent, but this function can also be used for deep Q-learning
    or DDPG, i.e. off-policy.

    If `discrete_actions` is true, the Q function is assumed to map states
    to action value arrays; if `discrete_actions` is false, the Q function
    is assumed to map state-action value pairs to corresponding value estimates.

    Params
    ======
    rewards (Tensor): reward(s) for taking current action(s)
    next_states (Tensor): state(s) reached after taking current action(s)
    Q (nn.Module): Q module with properties described above
    policy (nn.Module): policy derived from Q (e.g. `EpsilonGreedyPolicy`)
    gamma (float): discount factor
    discrete_actions (bool): determines the action space type    
    """
    
    next_actions = policy.act(next_states)
    if discrete_actions:
        # The policy may have collapsed dimensions by one (i.e. passing a
        # single state returns a scalar action, passing an array of states
        # returns a 1D array), so we may need to unsqueeze to prepare for gather
        if len(next_actions.shape) < len(next_states.shape):
            next_actions = next_actions.unsqueeze(-1)
        next_values = Q(next_states).gather(-1, next_actions)
        next_values = next_values.squeeze() # collapse dimensions again
    else:
        next_values = Q(next_states, next_actions)
    return rewards + gamma * next_values

def sarsamax_estimate(rewards, next_states, Q, gamma=1.):
    """
    Sarsamax estimator of the value of the current state-action pair. See
    docstring for `sarsa_estimate` for further details. This is similar
    except it doesn't need a policy. It also only works on discrete action
    spaces, so assumes the Q function maps states to action value arrays.
    """
    return rewards + gamma * torch.max(Q(next_states), dim=-1)

def expected_sarsa_estimate(rewards, next_states, Q, policy, gamma=1.):
    """
    Expected Sarsa estimator of the value of the current state-action pair.
    See docstring for `sarsa_estimate` for further details.

    The policy is expected to be a StochasticPolicy that generates Categorical
    distributions when the `stoch_act` method is called.
    """
    pi = policy.stoch_act(next_states).probs
    expected_next_values = torch.sum(pi * Q(next_states), dim=-1)
    return rewards + gamma * expected_next_values

def td0_estimate(rewards, next_states, V, gamma=1.):
    """
    TD(0) estimate of current state's value given observations of reward and
    next state.

    Works on single observations or batches.

    Params
    ======
    rewards (Tensor): reward(s) for taking current action(s)
    next_states (Tensor): next state(s) after taking current action(s)
    V (nn.Module): state value function
    gamma (float): discount factor
    """
    return rewards + gamma * V(next_states)
