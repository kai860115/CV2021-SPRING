import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # TODO
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

    def name(self):
        return "ConvNet"

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # TODO
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5),
            nn.BatchNorm2d(120),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )


    def forward(self, x):
        # TODO
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

    def name(self):
        return "MyNet"

