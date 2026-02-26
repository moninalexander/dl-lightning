from torch import nn

class MNIST_CNN_21(nn.Module):
    def __init__(self, ch0, im_size, ch1, ch2, num_out):
        super().__init__()
        self.box = nn.Sequential(
            nn.Conv2d(ch0,ch1,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(ch1,ch2,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(int(((im_size+1)/2+1)/2)**2*ch2,num_out)
        )
    def forward(self, x):
        logits = self.box(x)
        return logits
