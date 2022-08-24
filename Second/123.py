import numpy as np


def Prepare_Sum_proficiency(proficiency, state):
    Sum_proficiency = [a * b for a, b in zip(proficiency, state)]
    Sum_proficiency = np.sum(Sum_proficiency, axis=0)
    return Sum_proficiency


Process_time = [5, 10, 8]
Proficiency = [[0.87, 0.23, 0.64],
               [0.22, 0.33, 0.50],
               [0.91, 0.90, 0.14],
               [0.09, 0.78, 0.89],
               [0.23, 0.47, 0.81],
               [0.67, 0.99, 0.54]]
Worker1 = [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [1, 0, 1],
           [0, 0, 1],
           [0, 1, 0]]

Worker2 = [[1, 0, 0],
           [0, 0, 1],
           [0, 0, 1],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0]]

Proficiency_now1 = []
Proficiency_now2 = []
Utilization_rate1 = []
Utilization_rate2 = []
Proficiency_sum = np.sum(Proficiency, axis=1)
Worker_place1 = np.argmax(Worker1, axis=1)
Worker_place2 = np.argmax(Worker2, axis=1)
for i in range(3):
    Proficiency_now1.append(Proficiency[i][Worker_place1[i]])
    Proficiency_now2.append(Proficiency[i][Worker_place2[i]])
for i in range(3):
    Utilization_rate1.append(Proficiency_now1[i] / Proficiency_sum[i])
    Utilization_rate2.append(Proficiency_now2[i] / Proficiency_sum[i])
Utilization_rate1 = sum(Utilization_rate1)
Utilization_rate2 = sum(Utilization_rate2)
print(Utilization_rate1)
print(Utilization_rate2)
t1 = 0
t2 = 0
t_1 = Prepare_Sum_proficiency(np.array(Proficiency), Worker1)
t_2 = Prepare_Sum_proficiency(np.array(Proficiency), Worker2)
C = [a / b for a, b in zip(Process_time, t_1)]
D = [a / b for a, b in zip(Process_time, t_2)]
for i in range(3 - 1):
    t1 += max(C[0:i + 1]) + max(C[i:3])
    t2 += max(D[0:i + 1]) + max(D[i:3])
print(t1)
print(t2)
