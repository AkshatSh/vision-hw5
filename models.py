import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        if not os.path.exists('logs'):
            os.makedirs('logs')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open('logs/' + st, 'w')

    def log(self, str):
        print(str)
        self.logFile.write(str + '\n')

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001)

    def adjust_learning_rate(self, optimizer, epoch, args):
        lr = args.lr  # TODO: Implement decreasing learning rate's rules
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
       


class LazyNet(BaseModel):
    def __init__(self):
        super(LazyNet, self).__init__()
        # TODO: Define model here
        self.layer = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = x.view(-1, 32 * 32 * 3)
        x = self.layer(x)
        return x
        

class BoringNet(BaseModel):
    def __init__(self):
        super(BoringNet, self).__init__()
        # TODO: Define model here
        self.layers = nn.Sequential (
            nn.Linear(32 * 32 * 3, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        return self.layers.forward(x)


class CoolNet(BaseModel):
    def __init__(self):
        super(CoolNet, self).__init__()
        # TODO: Define model here
        self.firstLayer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.secondLayer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        self.lin1 = nn.Linear(2 * 2 * 64, 100)
        self.lin2 = nn.Linear(100, 10)

    def forward(self, x):
        # TODO: Implement forward pass for CoolNet
        x = self.firstLayer(x)
        x = self.secondLayer(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        return x
