import torch
import numpy as np
input_size = 10 # The number of expected features in the input x
hidden_size = 20 # THe number of features in the hidden state h

rnn = torch.nn.GRUCell(input_size, hidden_size)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)

print(output)