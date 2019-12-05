import torch
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(0, np.pi*2, 100), dim=1) # [100, 1]

# y = x.pow(3) + 0.1*torch.rand(x.size())
y = 3 * torch.sin(x)*torch.cos(2*x) + 0.3 * torch.rand(x.size())
# y = torch.sin(3*torch.cos(x)) + 0.2*torch.rand(x.size())
plt.scatter(x.numpy(), y.numpy())
plt.show()


def save():

    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    for i in range(5001):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    torch.save(net, 'net.pkl')
    torch.save(net.state_dict(), 'net_param.pkl')


def restore_net():
    net = torch.load('net.pkl')
    prediction = net(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    plt.show()


# save()
#
# restore_net()

