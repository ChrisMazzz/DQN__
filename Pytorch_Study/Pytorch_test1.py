x = [1, 0]
y = 0.5
w = [0.1, 0.2, 0.3]
lr = 0.01

def foward(x, y):
    out = x[0] * w[0] + x[1] * w[1] + w[2]
    return out

for i in range(100):
    out = foward(x, w)

    loss = (out - y) ** 2
    print(loss)
    td = [x[0] * 2 * (out - y), x[1] * 2 * (out - y), 2 * (out - y)]

    w[0] += -td[0] * lr
    w[1] += -td[1] * lr
    w[2] += -td[2] * lr

print(foward(x, w))