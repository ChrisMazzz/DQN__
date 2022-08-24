import numpy as np
import random as ran


class Schedul_env():
    action = 0
    r = 0

    def __init__(self):
        self.viewer = None
        # self.times = [10, 8, 15, 5, 5, 20, 10, 10, 3, 12, 15, 3, 8, 10, 12]
        self.times = [10, 8, 15, 5, 5]
        self.human = 15  # 实际上计算55个未分配的人就好
        self.maxcloths = 300
        # self.start_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # self.real_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.start_state = [1, 1, 1, 1, 1]
        self.real_state = [1, 1, 1, 1, 1]
        self.step_count = 0
        self.counts = 0
        # self.output_time = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # self.min_state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.output_time = [1, 1, 1, 1, 1]
        self.min_state = [1, 1, 1, 1, 1]
        self.actions = 5
        self.min = 99999
        self.done = 0
        self._r = 0
        # self.action_space = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.action_space = [1, 1, 1, 1, 1]

    # def time(self):
    #     t = 0
    #     division = [time / human for time, human in zip(self.times, self.real_state)]
    #     for i in range(self.actions - 1):
    #         t += max(division[0:i + 1])
    #         # print(t)
    #     for j in range(self.actions - 2):
    #         t += max(division[j:self.human - 1])
    #     t += (self.maxcloths - self.actions + 1) * max(division)
    #     # print(t)
    #     return t

    def time(self):
        t = 0
        C = [a / b for a, b in zip(self.times, self.real_state)]
        for i in range(self.actions - 1):
            t += max(C[0:i + 1]) + max(C[i:self.actions])
        t += (self.maxcloths - self.actions + 1) * max(C)
        return t

    def var(self):
        C = [a / b for a, b in zip(self.times, self.real_state)]
        t = np.var(C)
        return t

    def check_reward(self, done):
        reward = self.time()
        return reward

    def step(self, actions):
        a = actions
        if self.counts == self.human:
            r = -self.time()
            self.real_state[a] += 1
            self.counts += 1
            done = self.done
            if self.time() < self.min:
                r += 100
                self.min = self.time()
                self.min_state = self.real_state
            t = self.time()
            return self.real_state, r, done, t

        else:
            self.real_state[a] += 1
            done = 1
            r = self.min - self.check_reward(done)
            # self.real_state = self.start_state
            self.counts = 0
            x = done
            t = self.time()
            if self.time() < self.min:
                self.min = self.time()
                self.min_state = self.real_state
            print('------------------------------------------------')
            print(self.real_state)
            print(t)
            print('------------------------------------------------')
            print('\033[1;32m')  # 颜色绿色标注
            print('当前最佳时间为：')
            print(self.min)
            print('当前最佳情况为：')
            print(self.min_state)
            print('\033[0m')
            return self.real_state, r, x, t

    def reset(self):
        self.real_state = self.start_state
        self.done = 0
        self.counts = 0
        return self.real_state
