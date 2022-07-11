from torch.nn import Module
from torch import nn


class MyLeNet(Module):
    def __init__(self, dim_first_fn=2704):
        super(MyLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(dim_first_fn, 500)
        self.relu3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(500, 2)
        # self.relu4 = nn.LeakyReLU(0.1)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        # y = self.relu4(y)
        # y = self.fc3(y)
        return y