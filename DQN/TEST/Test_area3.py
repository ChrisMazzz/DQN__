import numpy as np
import random

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a * b
sum = 0
for i in range(3):
    print(c[i])
    sum += c[i]
a = random.randint(0, 15)
print(sum)
print(a)