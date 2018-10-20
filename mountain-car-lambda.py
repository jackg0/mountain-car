'''
Author: Jack Geissinger
Date: October 17, 2018

True Online Sarsa(lambda) implementation for Mountain Car

References: [1] Reinforcement learning: An introduction by RS Sutton, AG Barto, p. 252
            [2] R Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", NIPS 1996.
'''

import gym
import numpy as np
import math
from tiles3 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Initialize tile vector x and weight vector w
'''
x = np.zeros((4096,))
w = np.zeros((4096,))
iht = IHT(4096)

'''
Initialize constants used in algorithm.
alpha is the learning rate, gamma is the discount factor,
l (lambda) is the decay rate for the eligibility trace.
'''
alpha = 0.0125
gamma = 1
l = 0.9

'''
Initialize bounds of position and velocity.
'''
min_position, max_position = -1.2, 0.6
max_speed = 0.07

'''
decode takes in a state and turns in into indices which will modify x
'''
def decode(state, action):
    pos = state[0]
    vel = state[1]

    pos_scale = 8*pos//(max_position - min_position)
    vel_scale = 8*vel//(max_speed + max_speed)

    indices = tiles(iht, 8, [pos_scale, vel_scale], [action])

    return indices

'''
eps-greedy chooses the action which results in the maximum weight
'''
def eps_greedy(state):
    # init an array to store optional indices for greedy selection
    options = []
    # loop through possible actions to compare them all
    for possible_action in range(3):
        # possible actions are [-1, 0, 1], we want to get indices for each
        indices = decode(state, possible_action-1)
        # add indices to our options
        options.append(indices)
    # make an array of the approximate value function values to choose from
    values = [sum(w[indices]) for indices in options]
    # make a choice, choosing the max value
    choice = values.index(max(values))
    # get the best set of indices
    bestIndices = options[choice]
    # get the best action
    choices = [0, 1, 2]
    action = choices[choice]
    return bestIndices, action

'''
Actual algorithm implementation
'''
complete = 0
env = gym.make('MountainCar-v0').env
for i_episode in range(2000):
    # initialize state
    observation = env.reset()

    # take a new state and action greedily
    indices, action = eps_greedy(observation)

    # initialize x for a new episode
    x = np.zeros((4096,))
    x[indices] = 1

    # initialize eligibility trace for new episode
    z = np.zeros((4096,))

    Qold = 0

    for t in range(200):
        '''if i_episode > 1500:
            env.render()'''
        # Take action with new state and reward
        observation, reward, done, info = env.step(action)

        # greedily choose next action and state
        new_indices, action_p = eps_greedy(observation)

        # create new x' variable
        x_p = np.zeros((4096,))
        x_p[new_indices] = 1

        # modify Q and Q'
        Q = sum(w[indices])
        Q_p = sum(w[new_indices])

        delta = reward + gamma*Q_p - Q # updated error, discounting future prediction

        # modify eligibility trace
        z = gamma*l*z + (1 - alpha*gamma*l*sum(z[indices]))*x

        # update weights we are learning
        w = w + alpha*(delta + Q - Qold)*z - alpha*(Q - Qold)*x

        # modify scalar Qold
        Qold = Q_p

        # update x and action for next timestep
        x = x_p
        action = action_p
        indices = new_indices
        if done:
            print("Finished in {} timesteps".format(t+1))
            complete += 1
            break

    print("Finished episode {}".format(i_episode+1))
print("The number of successful trials: {}".format(complete))
