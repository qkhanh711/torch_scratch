
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.n_input = n_input
        self.fc1 = nn.Linear(n_input, 128)
        self.fc2 = nn.Linear(128, n_output)

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net2(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net2, self).__init__()
        self.n_input = n_input
        self.model = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        )

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = self.model(x)
        return x