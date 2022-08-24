import numpy as np
import time
import sys

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
EACH_TIME = [5, 6, 4, 6, 4]


class env:
    def __init__(self):
        self.n_actions = 5
        self.n_features = self.n_actions
        self.begin = [1, 1, 1, 1, 1]
        self.now_state = self.begin
        self.action_space = self.begin
        self.maxclothes = 300
        self.human_number = 20
        self.count = 0
        self.time_his = []  # 误差
        self.fancha_his = []
        self.done = False
        self.Min_time = 2000
        self.Max_time = 0
        self.Min_state = self.begin
        self.Min_episode = 0
        self.Min_episode_count = 0
        self.Error = False

    def step(self, action, episode):
        E = episode

        X = [0] * self.n_actions
        X[action] = 1
        self.now_state = [a + b for a, b in zip(self.now_state, X)]
        self.done = False
        if self.Variance() == 0:
            reward = 0
            self.Error = True
        else:
            reward = 1 / self.Variance()

        if sum(self.now_state) == self.human_number:
            reward = self.Check_reward()
            if E > 2000 and (self.Max_time + self.Min_time)/2 > self.Time():
                self.Error = True
            if self.Time() < self.Min_time:
                self.Cover(episode)
                reward += 1
            elif self.Time() == self.Min_time:
                self.Min_episode_count += 1
                if self.Min_episode_count >= 100:
                    E += 0.001
            self.Min_time = min(self.Time(), self.Min_time)
            print('\033[1;32m')  # 颜色绿色标注
            print(self.Time())
            print(self.now_state)
            print('\033[0m')
            E = self.Jump_out(E)  # 跳出局部最优
            self.done = True

        self.now_state = np.array(self.now_state)
        instead_state = self.now_state
        instead_time = self.Time()

        if sum(self.now_state) == self.human_number:
            self.now_state = self.begin
        return instead_state, reward, self.done, instead_time, E, self.Error

    def Check_reward(self):
        reward = 1 / self.Variance() + 0.01 * (self.Min_time - self.Time())
        reward -= 0.001 * self.Time()
        return reward

    def Time(self):
        t = 0
        C = [a / b for a, b in zip(EACH_TIME, self.now_state)]
        for i in range(self.n_actions - 1):
            t += max(C[0:i + 1]) + max(C[i:self.n_actions])
        t += (self.maxclothes - self.n_actions + 1) * max(C)
        return t

    def Variance(self):
        combat = [a / b for a, b in zip(EACH_TIME, self.now_state)]
        t = np.var(combat)
        return t

    def Cover(self, episode):
        self.Min_time = self.Time()
        self.Min_state = self.now_state
        self.Min_episode = episode
        self.Min_episode_count = 0

    def Jump_out(self, episode):
        E = episode
        if self.Time() == self.Min_time and self.Min_episode_count >= 1000:
            E += 0.5

        return E

    def reset(self):
        self.now_state = self.begin
        self.Error = False
        return self.begin

    def plot_cost(self):  # 误差曲线
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(np.arange(len(self.time_his)), self.time_his)
        plt.ylabel('time')
        plt.xlabel('steps')
        plt.show()
