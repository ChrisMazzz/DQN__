# coding = utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from numpy.random import rand
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

from Second.Schedul_env2 import env2

# parameters
Batch_size = 202
Lr = 0.01
Epsilon = 0.9  # greedy policy
Gamma = 0.9  # reward discount
Target_replace_iter = 100  # target update frequency
Memory_capacity = 2000
# env = gym.make('CartPole-v0')
# env = env.unwrapped
env = env2()
N_actions = 18
N_states = 18
N_Humans = 6
N_Lines = 3
# State = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# State = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# State = [[0, 0, 1],
#          [0, 1, 0],
#          [1, 0, 0],
#          [0, 0, 1],
#          [0, 1, 0],
#          [0, 1, 0]]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
Good_Count1 = 0
Good_Count2 = 0
episode = 20000
test_episode1 = 19500
test_episode2 = 9500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_actions, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.hide1 = nn.Linear(256, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, N_states)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.hide1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


def transform_s(state):
    new_s = np.array(state)
    new_s = np.squeeze(new_s, axis=0) if new_s.ndim == 2 else new_s
    return new_s


def transform_back(state):
    real_state = state.reshape(38, 5)
    return real_state


def store_transform(state):
    t_s = state.reshape(1, N_Humans * N_Lines)
    return t_s


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # initialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, epsilon):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()  # 网络输出最大单值
            print(action)
            # action = action_value.data.numpy()  # 网络输出整个数组
            # action = np.squeeze(action)
            # action = np.argmax(action)  # 行变化左边
            # real_action = np.argmax(action)
            # print(action)
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            # action_value = self.eval_net.forward(x)
            # action = torch.max(action_value, 1)[1].data.numpy()
            # action = action_value.data.numpy()
            # action = np.squeeze(action)

            action = np.random.randint(0, N_actions)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % Memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(Memory_capacity, Batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states + 1:N_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_net(self):  # 保存整个网络
        torch.save(self.eval_net, 'eval_net.pkl')
        torch.save(self.target_net, 'target_net.pkl')

    def restore_net(self):  # 提取整个网络
        # restore entire net1 to net2
        self.eval_net = torch.load('eval_net.pkl')
        self.target_net = torch.load('target_net.pkl')


if __name__ == '__main__':
    dqn = DQN()
    E = Epsilon
    print('\nCollecting experience...')
    for i_episode in range(episode):
        s = env.reset()
        while True:
            ep_r = 0
            s = transform_s(s)  # 矩阵转换
            a = dqn.choose_action(s, E)  # take action
            # a = a[0]
            _s = s
            s_, r, done, t, e, error = env.step(a, i_episode)
            s_ = transform_s(s_)  # 矩阵转换
            a = transform_s(a)
            if error:
                break
            s = _s
            dqn.store_transition(s, a, r, s_)
            ep_r += r
            if dqn.memory_counter > Memory_capacity:
                dqn.learn()
                # if done:
                #     print('Ep: ', i_episode,
                #           '| Ep_r: ', round(r))

            s = s_

            if i_episode > test_episode1 and done == 1:
                if t <= 450:
                    Good_Count1 += 1

            if i_episode > test_episode2 and done == 1:
                if t <= 450:
                    Good_Count2 += 1

            if done == 1:
                break

    print('\033[1;31m')
    print("中点的学习成果为：")
    print(Good_Count1 / 500)
    print("最终的学习成果为：")
    print(Good_Count2 / 500)
    print('\033[0m')
