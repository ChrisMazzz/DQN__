from ctypes import Array

import numpy
import numpy as np
import pandas as pd
import random as rand
import math
from random import choice
import time
import sys
from decimal import *  # 大数据下保持精度的方法

Process_time = [5, 10, 8]


class env2():

    def __init__(self):
        # super(Maze, self).__init__()  对继承父类属性进行初始化
        self.actions = 3
        self.Lines = 3
        self.maxclothes = 300
        self.Humans = 6
        self.count = 0
        self.done = False
        self.Min_time = 2000
        self.Max_time = 0
        self.Min_episode = 0
        self.Min_episode_count = 0
        self.action_space = [1, 1, 1]
        self.Error = False
        self.Have_Candidate = False
        self.Proficiency = np.zeros((self.Humans, self.Lines))
        self.Proficiency = [[0.87, 0.23, 0.64],
                            [0.22, 0.33, 0.50],
                            [0.91, 0.90, 0.14],
                            [0.09, 0.78, 0.89],
                            [0.23, 0.47, 0.81],
                            [0.67, 0.99, 0.54]]
        self.Proficiency = np.array(self.Proficiency)
        # self.Workers = [[0, 0, 1],
        #                 [0, 1, 0],
        #                 [1, 0, 0],
        #                 [0, 0, 1],
        #                 [0, 1, 0],
        #                 [0, 1, 0]]
        self.Workers = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        self.now_state = [[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [0, 1, 0]]
        self.Workers = np.array(self.Workers)
        self.begin = [[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [0, 1, 0]]
        self.Min_state = []
        self.Candidate = []
        self.false_action = []
        self.action_ = self.Humans * self.Lines + 1
        self.stay_count = 0

    def step(self, action, episode):
        E = episode
        self.Action_choice(action)
        self.count += 1
        self.done = False

        reward = self.Check_reward(action)
        if E > 2000 and (self.Max_time + self.Min_time) / 2 > self.Time():
            self.Error = True
        self.Min_time = min(self.Time(), self.Min_time)
        real_s = self.State_transform_back(self.now_state)
        E = self.Jump_out(E)  # 跳出局部最优
        if not self.Rule(self.now_state):
            self.now_state = self.Workers
        self.done = True
        self.now_state = np.array(self.now_state)
        instead_state = self.now_state
        instead_time = self.Time()
        x = self.State_transform(instead_state)  # 输出类型切换
        print(self.Rule(self.now_state))

        print('\033[1;32m')  # 颜色绿色标注
        print(self.Time())
        print(instead_state)
        print('\033[0m')

        return x, reward, self.done, instead_time, E, self.Error

    def Action_choice(self, a):
        act = np.array(a)
        if isinstance(act, list):  # 判断是否为数组
            act = a[0]
        actions_1 = act % 3
        actions_0 = act / 3
        x = math.floor(actions_0)
        y = actions_1
        for i in range(3):
            self.now_state[x][i] = 0
        if isinstance(y, numpy.ndarray):
            y = y[0]
        print(type(y))
        print(x, y)
        self.now_state[x][y] = 1
        if not self.Rule(self.now_state):
            self.now_state = self.begin

    # def Do_action(self, q_next, times):
    #     print(q_next)
    #     choice_worker = []
    #     cant_do = []
    #     action_list = []
    #     for i in range(times):
    #         print(cant_do)
    #         x = choice([j for j in range(0, 6) if j not in cant_do])
    #         cant_do.append(x)
    #         choice_worker.append(x)
    #     temporary_state = self.now_state
    #     for i in range(len(choice_worker)):
    #         for j in range(self.Lines):  # 提出随机抽选的工人的位置的q_next值
    #             action_list = q_next[choice_worker[i] * self.Lines + j]
    #         action = np.argmax(action_list)
    #         for j in range(self.Lines):
    #             temporary_state[choice_worker[i]][j] = 0
    #         temporary_state[choice_worker[i]][action] = 1
    #     judgement = self.Rule(temporary_state)
    #     if not judgement:
    #         self.false_action.append(choice_worker)
    #         times += 1
    #         self.Do_action(q_next, times)
    #     self.false_action = []
    #     return temporary_state

    def State_transform(self, state):
        new_s = []
        x = state.reshape(1, self.Lines * self.Humans)
        for i in x:
            s = list(map(int, i))
            new_s.append(s)
        return new_s

    def State_transform_back(self, state):
        s = np.array(state)
        x = s.reshape(self.Humans, self.Lines)
        return x

    def Check_reward(self, a):
        reward = 0
        Proficiency_sum = np.sum(self.Proficiency, axis=1)
        Worker_place = np.argmax(self.Workers, axis=1)
        if a == self.action_:  # 长期待在同一位置的惩罚
            self.stay_count += 1
            reward -= (1.0001 ** self.stay_count)
            print("*******", reward)
        self.action_ = a
        if not self.Rule(self.now_state):
            print(self.Rule(self.now_state))
            reward -= 10
            self.now_state = self.Workers
        Proficiency_now = []
        Utilization_rate = []
        for i in range(self.Humans):
            Proficiency_now.append(self.Proficiency[i][Worker_place[i]])
        for i in range(self.Humans):
            Utilization_rate.append(Proficiency_now[i] / Proficiency_sum[i])
        Utilization_rate = sum(Utilization_rate)
        # print(Utilization_rate)
        # print(self.Variance())
        V = self.Variance()
        if str(1 / self.Variance()) == "nan":
            V = 99999999999999999
        b = Utilization_rate
        c = 1 / V
        d = self.Min_time - self.Time()

        b = round(b, 5)
        c = round(c, 5)
        d = round(d, 5)
        print(reward)
        print(b)
        print(c)
        print(d)
        print(0.01 * round(self.Time(), 5))
        reward += b + c + 2 * d
        reward -= 0.01 * round(self.Time(), 5)
        print(reward)
        return reward

    def Prepare_Sum_proficiency(self):
        if self.Rule(self.now_state):
            Sum_proficiency = [a * b for a, b in zip(self.Proficiency, self.now_state)]
        else:
            Sum_proficiency = [a * b for a, b in zip(self.Proficiency, self.begin)]
        Sum_proficiency = np.sum(Sum_proficiency, axis=0)
        return Sum_proficiency

    def Time(self):
        t = 0
        t_ = self.Prepare_Sum_proficiency()
        C = [a / b for a, b in zip(Process_time, t_)]
        for i in range(self.actions - 1):
            t += max(C[0:i + 1]) + max(C[i:self.actions])
        return t

    def Variance(self):
        x = self.Prepare_Sum_proficiency()
        combat = [a / b for a, b in zip(Process_time, x)]
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
        s = np.array(self.now_state)
        s_ = s.reshape(1, self.Humans * self.Lines)
        self.Error = False
        return s_

    def Rule(self, state):
        Count_1_Workers = np.sum(state, axis=0)
        Rule_judgment = True
        for i in range(self.Lines):
            if Count_1_Workers[i] == 0:
                Rule_judgment = False
        return Rule_judgment

    # def Exchange(self):
    #     while Have_Candidate:
    #         rand.choice(self.Candidate)
    #         a = 1  # 想要替换的数字
    #         b = 0  # 替换后的数字
    #         index = (self.Workers == a)
    #         self.Workers[index] = b
    #         Have_Candidate = False
