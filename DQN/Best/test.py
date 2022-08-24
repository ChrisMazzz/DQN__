import random
import torch
import torch.nn as nn
import numpy as np

from DQN.TEST.test_area2 import Schedul_env
from Schedul_env import env


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 15)
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        return self.fc(inputs)


env2 = env()
env = Schedul_env()
net = MyNet()  # 主网络，保证学习效率
net2 = MyNet()  # 延迟更新，过一段时间更新目标预测，为目标网络
net3 = MyNet()  # 逐步网络，非结果网络，用来更新步骤

store_count = store_count2 = 0  # 经验池，记录状态环境，遇到了多少个情况
store_size = 1000  # 经验池的存储数据大小,此时表示可以记录2000个数据
decline = 0.4  # 随机衰减的系数，一开始会按照自己的想法随机选择运作，但随着学习的强化，就会变得越来越符合预设方向
learn_time = learn_time2 = 0  # 学习的次数记录
update_time = 500  # 学习多少次后会更新目标网络
gamma = 0.9  # 预测的衰减率
b_size = c_size = 500  # 记忆库的回响，表示学习了n次后，会从记忆库store中提取多少条数据参与学习
store = np.zeros((store_size, 33))  # 初始化记忆库，10为全参数(s,a,s_,r)的存储空间，每一个参数中有多少个记录参数，大小就为多少，详细看笔记
store2 = np.zeros((store_size, 33))
start_study = start_study2 = False  # 哨兵，用来提醒和显示什么时候开始学习
s = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
epoch = 10000
t = 0
_s = 0

for i in range(epoch):
    print(i)
    j = 0
    while True:
        j += 1
        if random.randint(0, 100) < 100 * (decline ** learn_time):  # 随机执行状态的选择，随着学习次数的增加，随机选择率会衰减
            a = random.randint(0, 14)  # 随机动作选择

        else:
            net.parameters()
            out = net(torch.Tensor(s)).detach()  # 如果没有随机学习，就会相信网络的学习，进行网络传值，detach()为反向传播的风险侦测
            a = torch.argmax(out).data.item()  # 动作选择，贪心选择奖励高的动作，argmax()函数为将reward输入进去，提取最大值的索引
            # 语句的返回值原理为从元组中选择出奖励最高的动作，元组的构成形式为(值, 最大值的索引)
            # 最大值索引中提取动作进行赋值，原本都是Tensor格式，所以用data转值后取出，item为提取索引中的项


        store[store_count % store_size][0:15] = s  # 记忆库如果满了，就进行覆盖
        _s = s
        s_, r, done = env2.step(a)  # 把我们选择的动作告诉环境，环境返回完成动作后的参数


        store[store_count % store_size][15:16] = a
        store[store_count % store_size][16:31] = s_
        store[store_count % store_size][31:32] = r
        store_count += 1
        if done == 1:
            s = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        s = s_
        # else:
        #     store2[store_count % store_size][0:15] = s  # 记忆库如果满了，就进行覆盖
        #     store2[store_count % store_size][15:16] = a
        #     store2[store_count % store_size][16:31] = s_
        #     store2[store_count % store_size][31:32] = r
        #     store2[store_count % store_size][32:33] = done
        #     store_count2 += 1
        #     s = s_
        #
        # if store_count2 > store_size:  # 若记忆库超过最大上限，学习次数超过阈值
        #
        #     if learn_time2 % update_time == 0:
        #         net3.load_state_dict(net.state_dict())  # 将此时的权重更新到目标网络的权重之中
        #
        #     index = random.randint(0, store_size - c_size - 1)  # 随机取一个位置，从里面去取1000条
        #     c_s = torch.Tensor(store2[index:index + c_size, 0:15])
        #     c_a = torch.Tensor(store2[index:index + c_size, 15:16]).long()
        #     c_s_ = torch.Tensor(store2[index:index + c_size, 16:31])
        #     c_r = torch.Tensor(store2[index:index + c_size, 31:32])
        #     c_d = torch.Tensor(store2[index:index + c_size, 32:33])
        #     q = net(c_s).gather(1, c_a)  # net(b_s)输出了[r1, r2],gather(1, b_a)表示对第一个维度进行聚合,聚合只留下索引中做当前动作所留下的预期奖励
        #     q_next = net3(c_s_).detach().max(1)[0].reshape(c_size, 1)  # 后续奖励，要用滞后的网络进行推断，detach()是截断梯度流，
        #     # max()表示取第一维中的最大值，通俗认为是该行的最大值
        #     # reshape()是重塑形状的函数
        #     tq = c_r + gamma * q_next  # 添加预测衰减系数
        #     loss2 = net.mls(q, tq)
        #     net.opt.zero_grad()
        #     loss2.backward()
        #     torch.nn.init.xavier_uniform(q_next, gain=1)  # 均匀初始化
        #     net.opt.step()
        #     learn_time2 += 1
        #     if not start_study2:
        #         print('Start Study')
        #         start_study2 = True
        #         break

        if store_count > store_size:  # 若记忆库超过最大上限，学习次数超过阈值

            if learn_time % update_time == 0:
                net2.load_state_dict(net.state_dict())  # 将此时的权重更新到目标网络的权重之中

            index = random.randint(0, store_size - b_size - 1)  # 随机取一个位置，从里面去取1000条
            b_s = torch.Tensor(store[index:index + b_size, 0:15])
            b_a = torch.Tensor(store[index:index + b_size, 15:16]).long()
            b_s_ = torch.Tensor(store[index:index + b_size, 16:31])
            b_r = torch.Tensor(store[index:index + b_size, 31:32])

            q = net(b_s).gather(1, b_a)  # net(b_s)输出了[r1, r2],gather(1, b_a)表示对第一个维度进行聚合,聚合只留下索引中做当前动作所留下的预期奖励
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1)  # 后续奖励，要用滞后的网络进行推断，detach()是截断梯度流，
            # max()表示取第一维中的最大值，通俗认为是该行的最大值
            # reshape()是重塑形状的函数
            tq = b_r + gamma * q_next  # 添加预测衰减系数
            loss = net.mls(tq, q)
            net.opt.zero_grad()
            loss.backward()
            # with torch.no_grad():
            #     weights -= weights.grad * lr
            #     bias -= bias.grad * lr
            #     weights.grad.zero_()
            #     bias.grad.zero_()
            net.opt.step()

            learn_time += 1
            if not start_study:
                print('Start Study')
                start_study = True
                break

        if done:
            break
