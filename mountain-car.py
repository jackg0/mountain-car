'''
Author: Jack Geissinger
Date: October 15, 2018

References: [1] Reinforcement learning: An introduction by RS Sutton, AG Barto, ch. 10.1, p. 198
            [2] R Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", NIPS 1996.
'''

import gym
import numpy as np
import math
from tiles3 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Initialize tile vector x and weight vector w, and iht for tiling.
'''
x = np.zeros((4096,))
w = np.zeros((4096,))
iht = IHT(4096)

'''
Initialize vector for plotting weights
'''
X, Y = np.meshgrid(range(64), range(64))
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

'''
Initialize constants used in algorithm.
alpha is the learning rate, gamma is the discount factor.
'''
alpha = 0.01
gamma = 0.5

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
env = gym.make('MountainCar-v0').env
for i_episode in range(1000):
    observation = env.reset()
    indices, action = eps_greedy(observation)
    x = np.zeros((4096,))
    x[indices] = 1
    for t in range(500):
        if i_episode > 500:
            env.render()
        observation, reward, done, info = env.step(action)

        if done:
            w = w + alpha*(reward - sum(w[indices]))*x
            print("woohoooo!")
            print("Finished in {} timesteps".format(t+1))
            break

        old_indices = indices
        indices, action = eps_greedy(observation)
        #print("Here are the indices -> {} and the action -> {}".format(indices, action)

        w = w + alpha*(reward + gamma*sum(w[indices]) - sum(w[old_indices]))*x

        x = np.zeros((4096,))
        x[indices] = 1

    print("Finished episode {}".format(i_episode+1))

#ha.plot_surface(X, Y, w)
#plt.savefig('sarsa-weights_episode-{}.png'.format(i_episode+1))
