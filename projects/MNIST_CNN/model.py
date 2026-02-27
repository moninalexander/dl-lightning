import torch
from torch import nn

class MNIST_CNN_21(nn.Module):
    def __init__(self, ch0, im_size, ch1, ch2, num_out):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(ch0,ch1,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch1,ch2,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        with torch.no_grad():
            X = torch.randn(1,ch0, im_size,im_size)
            self.last_numel = self.feature_extractor(X).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.last_numel,num_out)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits
