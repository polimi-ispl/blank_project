import os

import src
import src.torch as arch
from params import batch_size, epochs, trained_models_root, model_name_torch

import pandas
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def train_loop(dataloader, model, loss_fn, optimizer, platform):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Move data on GPU
        X = X.to(platform)
        y = y.to(platform)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn, platform):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # Move data on GPU
            X = X.to(platform)
            y = y.to(platform)
            # Prediction on the validation set
            pred = model(X)
            val_loss += loss_fn(pred, y.type(torch.long)).item()
            correct += (pred.argmax(1) == y).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return correct, val_loss


def main():
    # Output path
    history_path = os.path.join(trained_models_root, '{:s}'.format(model_name_torch), 'history_torch.pt')

    os.makedirs(history_path, exist_ok=True)
    
    # select the computation device automatically:
    src.set_gpu(-1)
    # or manually the first GPU (None for CPU):
    # src.set_gpu(0)
    
    # set backend here to create GPU processes
    src.torch.set_backend()
    src.torch.set_seed()
    
    # define the computation platform for torch:
    platform = src.torch.platform()
    dtype = src.torch.dtype()

    # Transform to tensor and normalize to [0, 1]
    transform = transforms.Compose(
        [transforms.ToTensor()])

    # Load training and validation set, initialize Dataloaders
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    valset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = arch.model_torch()
    # Move the model on gpu either with
    model = model.to(platform)
    
    # define the optimization parameters
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Scheduler for reducing the learning rate on plateau
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)

    best_acc = 0
    no_improvement = 0     # n of epochs with no improvements
    patience = 10          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []
    for t in range(epochs):

        print(f"Epoch {t+1}\n-------------------------------")
        # Model training
        train_loop(train_dataloader, model, loss_fn, optimizer, platform)
        accuracy, val_loss = val_loop(val_dataloader, model, loss_fn, platform)
        lr_scheduler.step(val_loss)    # call the scheduler to reduce the lr if val loss is in plateau
        history.append({"epoch": t,
                        "loss": val_loss,
                        "lr": optimizer.param_groups[0]['lr']})
        # MODEL CHECKPOINT CALLBACK
        if accuracy > best_acc:
            # Callback for weight saving
            torch.save(model.state_dict(), history_path)
            best_acc = accuracy

        # EARLY STOP CALLBACK
        if val_loss < min_val_loss:   # Improvement in the new epoch
            no_improvement = 0
            min_val_loss = val_loss
        else:                           # No improvement in the new epoch
            no_improvement += 1
            
        if t > 5 and no_improvement == patience:    # Patience reached
            print(f'Early stopped at epoch {t}')
            # Save history for early stopping
            df = pandas.DataFrame(history)
            df.to_csv("history.csv")
            break

    print("Done!")
    # Save history
    df = pandas.DataFrame(history)
    df.to_csv("history.csv")


if __name__ == '__main__':
    main()
