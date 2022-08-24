import torch
import torch.nn as nn
from torch.autograd import Variable

# y = x ** 2
# x = [3]
# x = Variable(torch.Tensor(x),requires_grad = True)
# y = x**2
# y.backward()
# print(x.grad)
# print(x.data)

# x = [[0.1, 0.8, 1], [0.8, 0.2, 1]]
# y = [[1], [0]]
# w = [[0.1, 0.2, 0.3]]
# x = Variable(torch.Tensor(x), requires_grad = False)
# y = Variable(torch.Tensor(y), requires_grad = False)
# w = Variable(torch.Tensor(w), requires_grad = True)
#
# for i in range(100):
#     out = torch.mm(x, w.t()) #w.t()为矩阵的转置操作
#     delta = (out - y)
#     loss = delta[0] ** 2 + delta[1] ** 2
#     print(loss)
#     w.grad = torch.Tensor([[0, 0, 0]]) #反向传播前要先归零，梯度是一个累加的过程，如果方向传播不先归零，就会出现重复累加的现象
#     loss.backward()
#     w.data -= w.grad * 0.01
# print(torch.mm(x, w.t()))
x = Variable(torch.Tensor([[0.1, 0.8], [0.8, 0.2]]))
y = Variable(torch.Tensor([[1], [0]]))

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer = nn.Linear(2, 1) #定义有多少个输入，多少个输出，分别对应（）内的第一个和第二个参数，若输出为2，则是两个神经元共享两个输入，但是分别输出两个数据
        self.layer2 = nn.Linear(1, 2) #除了输入层和输出层之外，中间的叫做隐藏层
        self.layer3 = nn.Linear(2, 3) #输出层

    def forward(self, x):
        out = self.layer(x)
        out = torch.Relu(out) #激活函数，用来判断是否需要继续激活层
        out = self.layer2(out)
        out = self.layer3(out)
        return out

net = MyNet()
mls = nn.MSELoss() #MSELoss为nn中的一个计算损失函数的方法，计算损失函数定义为如何去计算误差
opt = torch.optim.Adam(net.parameters(), lr=0.01)
#optim为torch中的计算优化器，Adam是集合了各种优缺点的优化器，没有特殊要求选这个即可，里面最重要的参数就是写明对谁进行优化--net.parameters(),其中包含了许多权重信息
print(net.parameters())
for i in range(1000):
    out = net(x) #放入神经网络进行计算
    loss = mls(out, y) #计算机进行误差求解
    print(loss)
    opt.zero_grad() #调用函数进行梯度清零
    loss.backward()
    opt.step() #利用权重优化器进行权重更新

print(net(x))
