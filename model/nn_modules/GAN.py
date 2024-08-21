import torch
from torch import nn
import torch.nn.functional as F

class SimpleGAN(nn.Module):
    def __init__(self):
        super(SimpleGAN, self).__init__()
        self.Generator = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
        self.Discriminator = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 100)
        x = self.Generator(x)
        x = self.Discriminator(x)
        return x
    
class SimpleGAN2(nn.Module):
    def __init__(self):
        super(SimpleGAN2, self).__init__()
        self.Generator = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )
        self.Discriminator = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 100)
        x = self.Generator(x)
        x = self.Discriminator(x)
        return x
    
if __name__ == "__main__":
    n_input = 784
    n_output = 10
    x = torch.randn(1, n_input)
    net = SimpleGAN()
    out = net(x)
    assert out.size() == (1, n_output), f"Bad output shape: {out.size()}"

    net2 = SimpleGAN2()
    out = net2(x)
    assert out.size() == (1, n_output), f"Bad output shape: {out.size()}"

    print("Passed all tests!")

