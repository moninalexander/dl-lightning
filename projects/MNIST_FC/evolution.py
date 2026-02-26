import torch

def train_eval_run(regime, sample_size, data_loader, model, loss_fn, optimizer=None):
    if regime == 'train':
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
        return sum(batch_loss)/sample_size, correct/total
    elif regime == 'eval':
        model.eval()
        batch_loss = []
        correct = 0
        total = 0 
        with torch.no_grad():
            for batch, (X, y) in enumerate(data_loader):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                batch_loss.append((len(y)*loss).item())
                correct+=(torch.argmax(y_pred,1)==y).sum().item()
                total+=len(y)
        return sum(batch_loss)/sample_size, correct/total
    else:
        print(f"The parameter 'regime' has to be either 'train' or 'eval'")


    
        
    