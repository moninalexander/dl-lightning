from torch import nn
from data import loaders

class MNIST_FC2(nn.Module):
    def __init__(self, num_in, n1, n2, num_out):
        super().__init__()
        self.flatten = nn.Flatten()
        self.box = nn.Sequential(
            nn.Linear(num_in,n1),
            nn.ReLU(),
            nn.Linear(n1,n2),
            nn.ReLU(),
            nn.Linear(n2,num_out)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.box(x)
        return logits