import numpy as np

Proficiency = [[0.87, 0.23, 0.64],
               [0.22, 0.33, 0.50],
               [0.91, 0.90, 0.14],
               [0.09, 0.78, 0.89],
               [0.23, 0.47, 0.81],
               [0.67, 0.99, 0.54]]

Workers1 = [[1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1]]

Workers2 = [[0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0]]

Proficiency_now = []
Utilization_rate = []
Worker_place1 = np.argmax(Workers1, axis=1)
Worker_place2 = np.argmax(Workers2, axis=1)
Proficiency_sum = np.sum(Proficiency, axis=1)

for i in range(6):
    Proficiency_now.append(Proficiency[i][Worker_place1[i]])
for i in range(6):
    Utilization_rate.append(Proficiency_now[i] / Proficiency_sum[i])
Utilization_rate1 = sum(Utilization_rate)

for i in range(6):
    Proficiency_now.append(Proficiency[i][Worker_place2[i]])
for i in range(6):
    Utilization_rate.append(Proficiency_now[i] / Proficiency_sum[i])
Utilization_rate2 = sum(Utilization_rate)

print(Utilization_rate1)
print(Utilization_rate2)

