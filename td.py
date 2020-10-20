# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    a = random.random()
    if a <= epsilon:
        action = np.random.randint(0,nA)
    else:
        action = np.argmax(Q[state][:]) 

    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # define decaying epsilon
    epsilon_decay = lambda e: 0.99 * e
    # loop n_episodes
    for episode in range(n_episodes):
        # initialize the environment 
        c_state = env.reset()
        # get an action from policy
        c_action = epsilon_greedy(Q, c_state, env.nA, epsilon)
        epsilon = epsilon_decay(epsilon)
        # loop for each step of episode
        while True:
            # one step in the environment
            # return a new state, reward and done
            next_state, reward, done, info = env.step(c_action)
            # get next action
            next_action = epsilon_greedy(Q, next_state, env.nA, epsilon)
            # TD update
            # td_target
            td_target = reward + gamma * Q[next_state][next_action]
            # td_error
            td_error = td_target - Q[c_state][c_action]
            # new Q
            Q[c_state][c_action] = Q[c_state][c_action] + alpha * td_error
            if done:
                break
            # update state
            c_state = next_state
            # update action
            c_action = next_action

    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    for episode in range(n_episodes):
        # initialize the environment 
        c_state = env.reset()

        # loop for each step of episode
        while True:
            # get an action from policy
            c_action = epsilon_greedy(Q, c_state, env.nA, epsilon)
            # return a new state, reward and done
            next_state, reward, done, info = env.step(c_action)
            # get next action
            next_action = np.argmax(Q[next_state])
            # TD update
            # td_target
            td_target = reward + gamma * Q[next_state][next_action]
            # td_error
            td_error = td_target - Q[c_state][c_action]
            # new Q
            Q[c_state][c_action] = Q[c_state][c_action] + alpha * td_error
            if done:
                break
            # update state
            c_state = next_state

    return Q
