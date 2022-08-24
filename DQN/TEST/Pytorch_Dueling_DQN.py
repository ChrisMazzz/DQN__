# coding = utf-8

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from DQN.TEST.test_area2 import Schedul_env

# parameters
Batch_size = 5
Lr = 0.01
Epsilon = 0.95  # greedy policy
Gamma = 0.9  # reward discount
Target_replace_iter = 100  # target update frequency
Memory_capacity = 2000
# env = gym.make('CartPole-v0')
# env = env.unwrapped
env = Schedul_env()
env2 = env()
N_actions = 5
N_states = 5
# State = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
State = [1, 1, 1, 1, 1]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
ENV_A_SHAPE = env.action_space
Good_Count1 = 0
Good_Count2 = 0
episode = 20000
test_episode1 = 19500
test_episode2 = 9500


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out1 = nn.Linear(64, N_actions)
        self.out1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(N_states, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(64, N_actions)
        self.out2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = F.relu(x1)

        x2 = self.fc2(x)
        x2 = F.relu(x2)

        y1 = self.out1(x1)
        y2 = self.out2(x2)
        x3 = y1.expand_as(y2) + (y2 - y2.mean(1).expand_as(y2))

        actions_value = x3
        return actions_value


class Dueling_DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((Memory_capacity, N_states * 2 + 2))  # innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=Lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < Epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = [np.random.randint(0, N_actions)]
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
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


if __name__ == '__main__':
    dqn = Dueling_DQN()
    print('\nCollecting experience...')
    for i_episode in range(episode):
        s = env.reset()
        while True:
            ep_r = 0
            a = dqn.choose_action(s)
            # print(a)

            # take action
            a = a[0]
            _s = s
            s_, r, done, t = env2.step(a)
            s = _s
            dqn.store_transition(s, a, r, s_)
            ep_r += r
            if dqn.memory_counter > Memory_capacity:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(r))

            s = s_

            if i_episode > test_episode1 and done == 1:
                if t <= 450:
                    Good_Count1 += 1

            if i_episode > test_episode2 and done == 1:
                if t <= 450:
                    Good_Count2 += 1

            # print(t,  Good_Count,  done)

            if done == 1:
                break

    print(t)
    print('\033[1;31m')
    print("中点的学习成果为：：")
    print(Good_Count1 / 500)
    print("最终的学习成果为：")
    print(Good_Count1 / 500)
    print('\033[0m')
