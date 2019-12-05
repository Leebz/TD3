import torch
import numpy as np


np_data = np.arange(6).reshape((2, 3))
a_data = np.arange(6).reshape((3, 2))

torch_data = torch.from_numpy(np_data)
b_data = torch.from_numpy(a_data)

print(torch.mm(torch_data, b_data))

data = [1, 2, 3]
tdata = torch.IntTensor(data)
print(tdata.dot(tdata))