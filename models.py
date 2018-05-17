import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
    def __init__(self, log_dir):
        super(BaseModel, self).__init__()
        dir_name = os.path.join(log_dir, 'logs/')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S_log.txt')
        self.logFile = open(str(dir_name) + st, 'w')

    def log(self, str, shouldPrint=False):
        if shouldPrint:
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
    def __init__(self, log_dir, device):
        super(LazyNet, self).__init__(log_dir)
        # TODO: Define model here
        self.device = device
        self.layer = nn.Linear(32 * 32 * 3, 10).to(device)

    def forward(self, x):
        # TODO: Implement forward pass for LazyNet
        x = x.view(-1, 32 * 32 * 3)
        x = self.layer(x)
        return x
        

class BoringNet(BaseModel):
    def __init__(self, log_dir, device):
        super(BoringNet, self).__init__(log_dir)
        # TODO: Define model here
        self.device = device
        self.layers = nn.Sequential (
            nn.Linear(32 * 32 * 3, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        ).to(device)
    def forward(self, x):
        # TODO: Implement forward pass for BoringNet
        x = x.view(-1, 32 * 32 * 3)
        x =  self.layers.forward(x)
        return x


class CoolNet(BaseModel):
    def __init__(self,log_dir, device):
        super(CoolNet, self).__init__(log_dir)
        # TODO: Define model here
        self.device = device
        self.firstLayer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.secondLayer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        self.lin1 = nn.Linear(1024, 100))
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
