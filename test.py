import os

import torch


path = '/home/simonhilber/delete/2025_04_14-12_23_31'
file = 'cam_time_left.pt'

complete = os.path.join(path, file)
data = torch.load(complete)
deltas = []
for i in range(1, len(data)):
    delta = data[i] - data[i - 1]
    deltas.append(delta)
print(deltas)