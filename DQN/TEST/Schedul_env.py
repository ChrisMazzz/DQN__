"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the environment part of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
# EACH_TIME = [10, 8, 15, 5, 5, 20, 10, 10, 3, 12, 15, 3, 8, 10, 12]
EACH_TIME = [5, 6, 4, 6, 4]


class Maze():
    def __init__(self):
        # super(Maze, self).__init__()  对继承父类属性进行初始化
        # self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = 5
        self.n_features = 5
        self.begin = [1, 1, 1, 1, 1]
        self.now = [1, 1, 1, 1, 1]
        self.maxclothes = 300
        self.human_number = 20

    def reset(self):
        time.sleep(0.1)
        # return observation
        self.now = self.begin
        return np.array(self.begin)

    def time(self):
        t = 0
        C = [a / b for a, b in zip(EACH_TIME, self.now)]
        for i in range(self.n_actions - 1):
            t += max(C[0:i + 1]) + max(C[i:self.n_actions])
        t += (self.maxclothes - self.n_actions + 1) * max(C)
        return t

    def step(self, action):

        X = [0] * self.n_actions
        X[action] = 1

        if sum(self.now) != self.human_number:
            self.now = [a + b for a, b in zip(self.now, X)]
            reward = 0
            done = False
        else:
            if self.time() < 800:
                reward = 1
            elif self.time() < 700:
                reward = 2
            else:
                reward = -1
            # reward = self.time()*-1
            print(self.time())
            print(self.now)
            done = True

        self.now = np.array(self.now)
        return self.now, reward, done
