import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(0, np.pi*2, 100), dim=1) # [100, 1]

# y = x.pow(3) + 0.1*torch.rand(x.size())
y = 4*torch.sin(x)*torch.cos(3*x) + 0.2 * torch.rand(x.size())
plt.scatter(x.numpy(), y.numpy())
plt.show()

"""
Method 1:
"""
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = F.relu(self.hidden2(x))
#         x = self.predict(x)
#         return x
#
# net = Net(n_feature=1, n_hidden=10, n_output=1)

"""
Method 2：
"""
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()
for i in range(10001):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if i % 5==0:
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw=5)
        print(i, loss.data.numpy())
        # plt.text(0.5, 0, "Loss=%0.4f" % loss.numpy(), fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()




