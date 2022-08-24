# 非线性的层，卷积神经网络
import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            # 定义一个卷积层,设置步长进行逐一扫描卷积,如果是彩色图片就会有3个通道
            # 详细定义如下(输入通道数，输出通道的个数（卷积核的个数）,卷积核的大小（默认长款一致)，卷积移动步长，周边要填充多少个0(padding)\
            # padding的计算公式如下：padding = (k_size-1)/2
            nn.MaxPool2d(2),  # 二维的最大值池化，2为缩小一倍，64*128*128-->64*64*64
            nn.ReLU()
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  # 64*64*64 --> 128*64*64
            nn.MaxPool2d(2),  # 128*64*64-->128*32*32
            nn.ReLU()
        )
        self.fc = nn.Sequential(  # 全连接层
            nn.Linear(128 * 32 * 32, 64),  # 输入层，接住卷积层的数据
            nn.ReLU(),  # 激活函数,数据被映射到（-1,1）的范围
            nn.Linear(64, 48),  # 隐藏层
            nn.ReLU(),
            nn.Linear(48, 10)  # 输出层
        )  # nn模块中的一个容器，按照需要顺序调用

        def forward(self, inputs):
            out = self.con1(inputs)
            out = self.con2(out)
            out = out.view(out.szie(0), -1)  # 展开成一维的
            out = self.fc(out)


cnn = MyCNN()
cnn.cuda()

x = torch.tensor([11, 2])
x.cuda()  # 放在GPU上计算
x.cpu()  # 放在CPU上运算
