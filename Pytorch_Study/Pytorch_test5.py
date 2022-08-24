import random
import torch
import torch.nn as nn
import numpy as np
import gym


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        return self.fc(inputs)


env = gym.envs.make('CartPole-v1')
env = env.unwrapped
net = MyNet()  # 主网络，保证学习效率
net2 = MyNet()  # 延迟更新，过一段时间更新目标预测，为目标网络

store_count = 0  # 经验池，记录状态环境，遇到了多少个情况
store_size = 2000  # 经验池的存储数据大小,此时表示可以记录2000个数据
decline = 0.6  # 随机衰减的系数，一开始会按照自己的想法随机选择运作，但随着学习的强化，就会变得越来越符合预设方向
learn_time = 0  # 学习的次数记录
update_time = 20  # 学习多少次后会更新目标网络
gamma = 0.9  # 预测的衰减率
b_size = 1000  # 记忆库的回响，表示学习了n次后，会从记忆库store中提取多少条数据参与学习
store = np.zeros((store_size, 10))  # 初始化记忆库，10为全参数(s,a,s_,r)的存储空间，每一个参数中有多少个记录参数，大小就为多少，详细看笔记
start_study = False  # 哨兵，用来提醒和显示什么时候开始学习
for i in range(50000):
    s = env.reset()  # 重置环境最初始的状态，重置环境中所有的参数
    while True:
        if random.randint(0, 100) < 100 * (decline ** learn_time):  # 随机执行状态的选择，随着学习次数的增加，随机选择率会衰减
            a = random.randint(0, 1)  # 随机动作选择

        else:
            out = net(torch.Tensor(s)).detach()  # 如果没有随机学习，就会相信网络的学习，进行网络传值，detach()为反向传播的风险侦测
            a = torch.argmax(out).data.item()  # 动作选择，贪心选择奖励高的动作，argmax()函数为将reward输入进去，提取最大值的索引
            # 语句的返回值原理为从元组中选择出奖励最高的动作，元组的构成形式为(值, 最大值的索引)
            # 最大值索引中提取动作进行赋值，原本都是Tensor格式，所以用data转值后取出，item为提取索引中的项
        s_, r, done, info = env.step(a)  # 把我们选择的动作告诉环境，环境返回完成动作后的参数
        r = (env.theta_threshold_radians - abs(s_[2])) / env.theta_threshold_radians * 0.7 + (
                env.x_threshold - abs(s_[0])) / env.x_threshold * 0.3  # 奖励的设置
        store[store_count % store_size][0:4] = s  # 记忆库如果满了，就进行覆盖
        store[store_count % store_size][4:5] = a
        store[store_count % store_size][5:9] = s_
        store[store_count % store_size][9:10] = r
        store_count += 1
        s = s_

        if store_count > store_size:  # 若记忆库超过最大上限，学习次数超过阈值

            if learn_time % update_time == 0:
                net2.load_state_dict(net.state_dict())  # 将此时的权重更新到目标网络的权重之中

            index = random.randint(0, store_size - b_size - 1)  # 随机取一个位置，从里面去取1000条
            b_s = torch.Tensor(store[index:index + b_size, 0:4])
            b_a = torch.Tensor(store[index:index + b_size, 4:5]).long()
            b_s_ = torch.Tensor(store[index:index + b_size, 5:9])
            b_r = torch.Tensor(store[index:index + b_size, 9:10])

            q = net(b_s).gather(1, b_a)  # net(b_s)输出了[r1, r2],gather(1, b_a)表示对第一个维度进行聚合,聚合只留下索引中做当前动作所留下的预期奖励
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1)  # 后续奖励，要用滞后的网络进行推断，detach()是截断梯度流，
            # max()表示取第一维中的最大值，通俗认为是该行的最大值
            # reshape()是重塑形状的函数
            tq = b_r + gamma * q_next  # 添加预测衰减系数
            loss = net.mls(q, tq)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()

            learn_time += 1
            if not start_study:
                print('Start Study')
                start_study = True
                break

        if done:
            break

        env.render()  # 渲染画面
