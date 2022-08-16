import torch
from torchstat import stat
from torch import nn

class Net(nn.Module):
    def __init__(self, dropout = 0.):
        super().__init__()
        self.fc = nn.Linear(10,10)
        # self.w1 = nn.Parameter(torch.ones(1, 10, 100))
        # self.w2 = nn.Parameter(torch.ones(1,100, 1000))
        self.w1 = torch.ones(1, 10, 100)
        self.w2 = torch.ones(1,100, 1000)
        self.w3 = torch.ones(1, 1, 1024, 10)
    def forward(self, x):
        x = self.fc(torch.cat([x,self.w3],dim=2))
        x = torch.matmul(x,self.w1)
        x = torch.matmul(x,self.w2)
        return x

net = Net()
stat(net, (1,256,10))
