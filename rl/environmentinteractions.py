import numpy as np
from time import time

def play_unity_episode(env, agent, train_mode, max_steps=10000,
                       states_transformer=None,
                       actions_transformer=None,
                       rewards_transformer=None):
    """
    Play a single episode of a Unity environment using the given agent. The
    terminates when any agent (in a multi-agent environment) is done.

    We assume agent exposes the following methods:
    - act(states): return intended actions for a batch of states
    - observe(states, actions, rewards, next_states, dones): observe outcomes
      from the environment and (if they wish) learn from these
    - save(filename): save agent learned parameters to file
    - load(filename): load agent learned parameters from file

    We assume the environment conforms to a Unity ML-Agents API, with a
    single brain.

    Params
    ======
    env: Unity-style environment
    agent (Agent): the agent object
    train_mode (bool): sets training mode on or off for both env and agent.
    max_steps (int): maximum number of steps to take before terminating episode
    states_transformer (None or callable): apply a transformation to the states
        vector received from the environment before passing on to the agent.
    actions_transformer (None or callable): apply a transformation to the
        actions vector received from the agent before passing to the
        environment.
    rewards_transformer (None or callable): apply a transformation to the
        rewards vector received from the environment before passing on to the
        agent.

    Returns
    =======
    scores (list): total score per agent in the environment
    """
    if states_transformer is None:
        states_transformer = lambda x:x
    if actions_transformer is None:
        actions_transformer = lambda x:x
    if rewards_transformer is None:
        rewards_transformer = lambda x:x
    
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=train_mode)[brain_name]
    states = states_transformer(env_info.vector_observations)
    num_agents = len(env_info.agents)
    steps = 0
    scores = np.zeros(num_agents)
    dones = [False] * num_agents
    while steps < max_steps and not np.any(dones):
        actions = agent.act(states)
        env_info = env.step(actions_transformer(actions))[brain_name]
        next_states = states_transformer(env_info.vector_observations)
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        agent.observe(states, actions, rewards_transformer(rewards),
                      next_states, dones)
        states = next_states
        steps += 1
    return scores, steps

def train_agent(env, agent, num_episodes, weights_file,
                max_steps=10000, target_average_score=30,
                over_how_many_eps=100, score_aggregator=np.mean,
                states_transformer=None, actions_transformer=None,
                rewards_transformer=None, new_line_every=10,
                previous_scores=None):
    """
    Train the agent. For this function `agent` should have a `LearningAgent`
    interface.

    Params
    ======
    num_episodes (int): maximum number of training episodes
    weights_file (str): filename for saving agent's learned parameters
    target_average_score (float): average score for considering env solved
    over_how_many_eps (int): how many episodes to calculate ave. over
    score_aggregator (callable): how to aggregate per episode scores over
        multiple agents
    new_line_every (int): how often to print a new line showing training
        progress
    previous_scores (list): result of previous run (to continue training)
    [Other parameters as per play_episode]

    Returns
    =======
    scores (list): List of average scores from training episodes
    """
    scores = [] if previous_scores is None else previous_scores
    steps = len(scores)
    for i_episode in range(1, num_episodes+1):#
        start_t = time()
        agent_scores, ep_steps = play_unity_episode(env, agent,
                                                    train_mode=True,
                                                    max_steps=max_steps,
                                                    states_transformer=states_transformer,
                                                    actions_transformer=actions_transformer,
                                                    rewards_transformer=rewards_transformer)
        step_rate = ep_steps / (time() - start_t)
        scores.append(score_aggregator(agent_scores))
        steps += ep_steps
        end_line = '\n' if (i_episode % new_line_every == 0) else ''
        running_ave_score = np.mean(scores[-over_how_many_eps:])
        print('\rEpisode {:4}/{:4} | Steps: {:6} | Steps/s: {:3.2f} | '
              'Ep score: {:2.2f} | '
              '{}-ave: {:2.2f} | {}-ave: {:2.2f}'
              ''.format(i_episode, num_episodes, steps, step_rate,
                        np.mean(agent_scores), new_line_every,
                        np.mean(scores[-new_line_every:]),
                        over_how_many_eps, running_ave_score),
              end=end_line)
        if i_episode >= over_how_many_eps and running_ave_score >= target_average_score:
            print('\nEnvironment solved!')
            break
    agent.save(weights_file)
    return scores, steps
