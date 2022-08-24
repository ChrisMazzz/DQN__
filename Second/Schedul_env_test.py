import numpy as np
import random as rand

lines = 5
Have_Candidate = False
Humans = 20

print(np.random.random(lines))

Proficiency = np.zeros((Humans, lines))

for i in range(20):
    Proficiency[i] = np.random.random(5)

print(Proficiency)

Workers = np.zeros((Humans, lines))

Workers = np.full((Humans, lines), 0, dtype=int)

for i in range(int(20)):
    index = np.random.choice(Workers.shape[1], 1, replace=False)
    Workers[i][index] = [1]

print('----------------------------')
print(Workers)

Count_1_Workers = np.sum(Workers, axis=0)

print('----------------------------')
print(Count_1_Workers)

Max_line = np.argmax(Count_1_Workers)

Candidate = []
for i in range(5):
    if Count_1_Workers[i] == 0:
        Candidate.append(i)
        Have_Candidate = True

while Have_Candidate:
    rand.choice(Candidate)
    a = 1  # 想要替换的数字
    b = 0  # 替换后的数字
    index = (Workers == a)
    Workers[index] = b
    Have_Candidate = False

print(Candidate)

Sum_time = [a * b for a, b in zip(Proficiency, Workers)]
Sum_time = np.sum(Sum_time, axis=0)
print(Sum_time)


def Action(j):
    ran = rand.randint(0, 20)
    _var = np.argmax(Workers[ran])
    Workers[ran][j] += 1
    Workers[ran][_var] -= 1


def Choice_worker():
    return 0
