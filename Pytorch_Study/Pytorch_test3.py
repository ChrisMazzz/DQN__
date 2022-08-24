#线性的层
import torch
import torch.nn as  nn

x = torch.Tensor([[0.2, 0.4], [0.2, 0.3], [0.3, 0.4]]) #输入
x.cuda()
y = torch.Tensor([[0.6], [0.5], [0.7]]) #标签

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 4), #输入层
            nn.ReLU(), #激活函数,数据被映射到（-1,1）的范围
            nn.Linear(4, 4), #隐藏层
            nn.ReLU(),
            nn.Linear(4, 1)  #输出层
        ) #nn模块中的一个容器，按照需要顺序调用
        self.opt = torch.optim.Adam(self.parameters()) #定义优化器
        self.mls = torch.nn.MSELoss() #定义损失函数，是函数的引用，并非直接调用

    def forward(self, inputs): #前向传播函数
        return self.fc(inputs)

    def train_model(self, x, y): #用输入和标签去训练我们的模型
        out = self.forward(x)
        loss = self.mls(out, y)
        print('loss：',loss)
        self.opt.zero_grad() #归零梯度
        loss.backward()
        self.opt.step() #利用优化器进行逐步优化

    def test(self, x):
        return self.forward(x)

def main():
    net = MyNet()
    for i in range(10000):
        net.train_model(x, y)
    # x = torch.Tensor([[0.5, 0.3]])
    out = net.test(x)
    print(out)

if __name__ == "__main__":
    main()




