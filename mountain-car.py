'''
Author: Jack Geissinger
Date: October 15, 2018

References: [1] Reinforcement learning: An introduction by RS Sutton, AG Barto, ch. 10.1, p. 198
            [2] R Sutton, "Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding", NIPS 1996.
'''

import gym
import numpy as np
import math

x = np.zeros((64, 64))
w = np.zeros((64, 64))

alpha = 0.5
gamma = 0.5

min_position, max_position = -1.2, 0.6
max_speed = 0.07

def decode(state):
    pos = state[0]
    vel = state[1]

    i = int(64*pos//(max_position - min_position))
    j = int(64*vel//(max_speed + max_speed))

    return (i, j)

def eps_greedy(state):

    options = []
    for possible_action in range(3):
        position, velocity = state
        velocity += (possible_action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -max_speed, max_speed)
        position += velocity
        position = np.clip(position, min_position, max_position)
        if (position==min_position and velocity<0): velocity = 0
        s_p = decode([position, velocity])
        options.append(s_p)
    values = [w[s_p[0], s_p[1]] for s_p in options]
    choice = values.index(max(values))
    s_p = options[choice]
    choices = [0, 1, 2]
    action = choices[choice]
    return s_p, action

env = gym.make('MountainCar-v0').env
for i_episode in range(1000):
    observation = env.reset()
    s_p, action = eps_greedy(observation)
    for t in range(1000):
        env.render()
        observation, reward, done, info = env.step(action)
        x = np.zeros((64, 64))
        (i, j) = decode(observation)
        x[i,j] = 1
        if done:
            w = w + alpha*(reward - w[i,j])*x
            print("woohoooo!")
            break

        s_p, action = eps_greedy(observation)

        w = w + alpha*(reward + gamma*w[s_p[0],s_p[1]] - w[i,j])*x
    print("Finished episode {}".format(i_episode+1))
