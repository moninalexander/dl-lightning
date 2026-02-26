import torch
from torch import nn
from data import loaders
from model import MNIST_CNN_21
import matplotlib.pyplot as plt

from torchinfo import summary

import time

BATCH_SIZE = 64
LEARNING_RATE = 0.001
CH1 = 32
CH2 = 64

train_loader, TRAIN_SIZE, test_loader, TEST_SIZE, IM_SIZE, CH0, NUM_CLASSES = loaders(batch_size=BATCH_SIZE)

model = MNIST_CNN_21(CH0, IM_SIZE, CH1, CH2, NUM_CLASSES)



summary(model, depth=3, input_size = (BATCH_SIZE, CH0, IM_SIZE, IM_SIZE))


J = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                       patience=2, factor=0.5, threshold=0.0)



def train_run(data_loader, model, loss_fn, optimizer):
    model.train()
    batch_loss = []
    correct = 0
    total = 0 
    for batch, (X, y) in enumerate(data_loader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss.append((len(y)*loss).item())
        correct+=(torch.argmax(y_pred,1)==y).sum().item()
        total+=len(y)
    return sum(batch_loss)/TEST_SIZE, correct/total


def test_run(data_loader, model, loss_fn):
    model.eval()
    batch_loss = []
    correct = 0
    total = 0 
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            batch_loss.append(loss.item())
            correct+=(torch.argmax(y_pred,1)==y).sum().item()
            total+=len(y)
    return sum(batch_loss)/len(batch_loss), correct/total


train_loss_history = []
test_loss_history= []

train_acc_history = []
test_acc_history = []



epochs = 20
print("=" * 70)
print(f"{'Epoch':<10}{'Model':<10}{'Test Accuracy':<20}{'Test Loss':<15}{'Learning Rate':<15}")
t1 = time.perf_counter()

best_acc = 0

for t in range(epochs):
    train_loss, train_acc=train_run(train_loader, model, J, optimizer)
    test_loss, test_acc=test_run(test_loader, model, J)
    scheduler.step(test_acc)

    print(f"{t:<10}{'CNN':<10}{test_acc:<20.4f}{test_loss:<15.4f}{scheduler.get_last_lr()[-1]:<15.6f}")
    
    test_acc_history.append(test_acc)
    test_loss_history.append(test_loss)
    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),"CNN.pt")
        print("-" * 70)
        print("The best model changed")
        print("-" * 70)
    
print("=" * 70)
t2 = time.perf_counter()
print(f"{'Time':<4}{t2-t1:>10.1f} s")





fig, ax = plt.subplots(2,2,figsize = (8,6),sharex=True)

ax[0,0].set_title('Test')
ax[0,1].set_title('Train')


ax[0,1].sharey(ax[0,0])
ax[1,1].sharey(ax[1,0])

fig.supxlabel("Epoch")



ax[0,0].set_ylabel('Accuracy')
ax[1,0].set_ylabel('Loss')

ax[0,0].plot(test_acc_history,'r.-',label="Convolutional")
# ax[0,0].legend()
ax[0,1].plot(train_acc_history,'r.-')


ax[1,0].plot(test_loss_history,'r.-')
ax[1,1].plot(train_loss_history,'r.-')

plt.tight_layout()
plt.show()