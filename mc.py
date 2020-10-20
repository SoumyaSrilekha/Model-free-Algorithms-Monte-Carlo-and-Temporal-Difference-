# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:12:31 2020

@author: Soumya Srilekha
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise
    Parameters:
    -----------
    observation
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    # action

    if observation[0] >= 20:
        
        action = 0;
        
    else:
        action = 1;
        
    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.
    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for episode in range(n_episodes):
        # initialize the episode
        state = env.reset()
        # generate empty episode list
        episode_list = []
        # loop until episode generation is done
        while True:
            # select an action
            action = policy(state)
            # return a reward and new state
            next_state, reward, done, info = env.step(action)
            # append state, action, reward to episode
            episode_list.append([state, action, reward])
            if done:
                break
            # update state to new state
            state = next_state


        # loop for each step of episode, t = T-1, T-2,...,0
        state_set = set()
        for ep in episode_list:
            state_set.add(ep[0])
        #Gt= Rt+1 *1 + Rt+2 * gamma + Rt+3 * gamma^2 +..
        for state in state_set:
            # first occurence of the observation
            # return the first index of state
            ind = episode_list.index([episode for episode in episode_list 
                                                  if episode[0] == state][0])
            # compute G
            Gt = sum([episode[2] * gamma ** i for i, episode in enumerate(episode_list[ind:])])           
            # update return_count
            returns_count[state] += 1.0
            # update return_sum
            returns_sum[state] += Gt
            # calculate average return for this state over all sampled episodes
            V[state] = returns_sum[state] / returns_count[state]

    ############################

    return V

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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
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

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

        # define decaying epsilon
    # epsilon_decay = lambda ep: ep - (0.1 / n_episodes)
    for episode in range(n_episodes):
        # initialize the episode
        epsilon = epsilon - (0.1 / n_episodes)
        state = env.reset()
        # generate empty episode list
        reward_list = []
        st_list = []
        pair = set()
        # loop until episode generation is done
        while True:
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            # return a reward and new state
            next_state, reward, done, info = env.step(action)
            # append state, action, reward to episode
            reward_list.append(reward)
            st_list.append((state, action))
            pair.add((state,action))
            if done:
                break
            # update state to new state
            state = next_state
            # epsilon = epsilon_decay(epsilon)
        # loop for each step of episode, t = T-1, T-2, ...,0
        # make a pair of state action 
        for state_action in pair:
            # first occurence of the observation
            # return the first index of state
            ind = st_list.index(state_action)
            # compute G
            Gt = sum([reward * gamma ** i for i, reward in enumerate(reward_list[ind:])])           
            # update return_count
            returns_count[state_action] += 1.0
            # update return_sum
            returns_sum[state_action] += Gt
            # calculate average return for this state over all sampled episodes
            Q[state_action[0]][state_action[1]] = returns_sum[state_action] / returns_count[state_action]

    return Q