import torch
from torch import nn

from torchinfo import summary

from data import loaders
from model import MNIST_FC2
from evolution import train_eval_run
from plot import plot_result

import time

import rich

##############################################################################################

BATCH_SIZE = 64
LEARNING_RATE = 0.001
N1=512
N2=512

train_loader, TRAIN_SIZE, test_loader, TEST_SIZE, IM_SIZE, CH0, NUM_CLASSES = loaders(batch_size=BATCH_SIZE)

model = MNIST_FC2(IM_SIZE**2, N1, N2, NUM_CLASSES)
best_model = MNIST_FC2(IM_SIZE**2, N1, N2, NUM_CLASSES)
summary(model, depth=3, input_size = (BATCH_SIZE, CH0, IM_SIZE, IM_SIZE))

J = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", 
                                                       patience=2, factor=0.5, threshold=0.0)

##############################################################################################

train_loss_history = []
test_loss_history= []

train_acc_history = []
test_acc_history = []


best_model.load_state_dict(torch.load('best_FC.pt'))
with torch.no_grad():
    _, best_acc=train_eval_run('eval', TEST_SIZE, data_loader=test_loader, model=best_model, loss_fn=J)
    rich.print(f"{'':<10}[green]{'Current best':<20}{best_acc:<15.4}")

epochs = 20
print("=" * 90)
print(f"{'Epoch':<10}{'Test Accuracy':<20}{'Test Loss':<15}{'Learning Rate':<15}")
print("=" * 90)
t1 = time.perf_counter()

for t in range(epochs):
    train_loss, train_acc=train_eval_run('train', TRAIN_SIZE, data_loader=train_loader, model=model, loss_fn=J, optimizer=optimizer)
    test_loss, test_acc=train_eval_run('eval', TEST_SIZE, data_loader=test_loader, model=model, loss_fn=J)
    scheduler.step(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(),"best_FC.pt")
        rich.print(f"[green]{t:<10}{test_acc:<20.4f}{test_loss:<15.4f}{scheduler.get_last_lr()[-1]:<15.6f}{'Current best is updated':<20}")
        # rich.print(f"{'':<10}[green]{best_acc:<20.4}{'':30}{'Current best is updated':<20}")
    else:
        print(f"{t:<10}{test_acc:<20.4f}{test_loss:<15.4f}{scheduler.get_last_lr()[-1]:<15.6f}")

    test_acc_history.append(test_acc)
    test_loss_history.append(test_loss)
    train_acc_history.append(train_acc)
    train_loss_history.append(train_loss)

    
    
print("=" * 90)
t2 = time.perf_counter()
rich.print(f"{'Time':<4}{t2-t1:>10.1f} s")

##############################################################################################

plot_result(test_acc_history, train_acc_history, test_loss_history, train_loss_history)

