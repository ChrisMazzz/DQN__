#自动求倒数的构造方法

x = [[0.1, 0.8],[0.8,0.2]]
y = [1, 0]
w = [0.1, 0.2, 0.3]
lr = 0.01

class add_node():
    def __init__(self, x, y):
        self.res = x + y
        self.grad_x = 1
        self.grad_y = 1

class cheng_node():
    def __init__(self, x, y):
        self.res = x * y
        self.grad_x = x
        self.grad_y = y

class pingfang_node():
    def __init__(self, x):
        self.res = pow(x, 2)
        self.grad_x = 2 * x

class cell():
    def __init__(self, x, w):
        self.grad = []
        self.res = 0 #输出线性默认为0
        for i in range(len(x)):
            self.res += x[i] * w[i] #有多少个维度的输入，就有多少个维度的权重
            self.grad.append(x[i]) #偏置，也就是固定的1 * x的i，其中append()为用于在列表尾添加新的对象
        self.res += w[len(x)]
        self.grad.append(1) #最后一个不一样，梯度固定为1

for i in range(10000):
    out0 = cell(x[0], w)
    out1 = cell(x[1], w)
    print('out', [out0.res, out1.res])
    d0 = add_node(out0.res, out1.res)
    pn0 = pingfang_node(d0.res)
    d1 = add_node(out0.res, out1.res)
    pn1 = pingfang_node(d1.res)
    an1 = add_node(pn0.res, pn1.res)
    print('loss', an1.res)
    print(' ')
    td0 = an1.grad_x * pn0.grad_x * d0.grad_x * out0.grad[0] + an1.grad_y * pn1.grad_x * d1.grad_x * out1.grad[0]
    td1 = an1.grad_x * pn0.grad_x * d0.grad_x * out0.grad[1] + an1.grad_y * pn1.grad_x * d1.grad_x * out1.grad[1]
    td2 = an1.grad_x * pn0.grad_x * d0.grad_x * out0.grad[2] + an1.grad_y * pn1.grad_x * d1.grad_x * out1.grad[2]
    w[0] -= td0 * lr
    w[1] -= td1 * lr
    w[2] -= td2 * lr