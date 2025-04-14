import os

import torch


path = '/home/i53/student/shilber/Downloads/download'
file = 'leader_time.pt'

complete = os.path.join(path, file)
data = torch.load(complete)
deltas = []

for i in range(1, len(data)):
    delta = data[i] - data[i - 1]
    deltas.append(delta)
print(deltas)
#print(data)