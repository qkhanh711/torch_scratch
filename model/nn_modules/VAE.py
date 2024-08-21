import torch
from torch import nn
import torch.nn.functional as F

class SimpleAutoEncoder(nn.Module):
    def __init__(self, n_input, n_output):
        super(SimpleAutoEncoder, self).__init__()
        self.n_input = n_input
        self.fc1 = nn.Linear(n_input, 128)
        self.fc2 = nn.Linear(128, n_output)
        self.fc3 = nn.Linear(n_output, 128)
        self.fc4 = nn.Linear(128, n_input)

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

class SimpleAutoEncoder2(nn.Module):
    def __init__(self, n_input, n_output):
        super(SimpleAutoEncoder2, self).__init__()
        self.n_input = n_input
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_output, 128),
            nn.ReLU(),
            nn.Linear(128, n_input)
        )

    def forward(self, x):
        x = x.view(-1, self.n_input)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, n_input, n_output):
        super(VAE, self).__init__()
        self.n_input = n_input
        self.fc1 = nn.Linear(n_input, 128)
        self.fc21 = nn.Linear(128, n_output)
        self.fc22 = nn.Linear(128, n_output)
        self.fc3 = nn.Linear(n_output, 128)
        self.fc4 = nn.Linear(128, n_input)

    def encode(self, x):
        x = x.view(-1, self.n_input)
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = F.relu(self.fc3(z))
        x = self.fc4(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
    
