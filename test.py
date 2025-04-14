import os

import torch


path = '/home/i53/student/shilber/Downloads/download(1)/no_save_and_sleep'
file = 'cam_time_top.pt'
second = 'leader_time.pt'

complete1 = os.path.join(path, file)
complete2 = os.path.join(path, second)
data1 = torch.load(complete1)
data2 = torch.load(complete2)

# def difference(data1, data2):
#     diffrences = []
#     for i in 

def delta (data):
    deltas = []
    for i in range(1, len(data)):
        delta = data[i] - data[i - 1]
        deltas.append(delta)
    print(deltas)
delta(data1)
#print(data)