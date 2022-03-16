import os
from argparse import ArgumentParser
import numpy as np
import pandas
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import src
import src.torch_utils as arch


def train_loop(dataloader, model, loss_fn, optimizer, platform):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        # Move data on GPU
        X = X.to(platform)
        y = y.to(platform)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= len(dataloader)
    correct /= size
    
    return correct, train_loss


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
    parser = ArgumentParser(description="An example of training a CIFAR classifier in pytorch")
    parser.add_argument("--outpath", type=str, required=False,
                        default="./data/trained_models/pytorch",
                        help="Results directory")
    parser.add_argument("--gpu", type=int, required=False, default=None,
                        help="Index of GPU to use (None for CPU, -1 for least used GPU)")
    parser.add_argument("--batch_size", type=int, required=False, default=32,
                        choices=[16, 32, 64],
                        help="Batch size")
    parser.add_argument("--epochs", type=int, required=False, default=10,
                        help="Max iterations number")
    parser.add_argument("--lr", type=float, required=False, default=1e-3,
                        help="Learning rate")
    args = parser.parse_args()
    
    # save args to outpath, for reproducibility
    os.makedirs(args.outpath, exist_ok=True)  # set to True to enable overwriting
    src.write_args(filename=os.path.join(args.outpath, "args.txt"),
                   args=args)
    model_path = os.path.join(args.outpath, "best_model.pth")
    history_path = os.path.join(args.outpath, "history.csv")
    
    # select the computation device:
    src.set_gpu(args.gpu)

    # set backend here to create GPU processes
    src.torch_utils.set_backend()
    src.torch_utils.set_seed()
    
    # define the computation platform for torch:
    platform = src.torch_utils.platform()

    # Transform to tensor and normalize to [0, 1]
    transform = transforms.Compose(
        [transforms.ToTensor()])

    # Load training and validation set, initialize Dataloaders
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    valset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_dataloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = arch.model_torch()
    # Move the model on gpu either with
    model = model.to(platform)
    
    # define the optimization parameters
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Scheduler for reducing the learning rate on plateau
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)

    best_acc = 0
    no_improvement = 0     # n of epochs with no improvements
    patience = 10          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []
    for t in range(args.epochs):

        print(f"Epoch {t+1}\n-------------------------------")
        # Model training
        train_acc, train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, platform)
        val_acc, val_loss = val_loop(val_dataloader, model, loss_fn, platform)
        
        # SCHEDULER CALLBACK
        lr_scheduler.step(val_loss)    # call the scheduler to reduce the lr if val loss is in plateau
        history.append({"epoch": t,
                        "val_loss": val_loss,
                        "loss": train_loss,
                        "val_score": val_acc,
                        "score": train_acc,
                        "lr": optimizer.param_groups[0]['lr']})
        
        # MODEL CHECKPOINT CALLBACK
        if val_acc > best_acc:
            # Callback for weight saving
            # torch.save(model.state_dict(), model_path)
            src.torch_utils.save_model(net=model, optimizer=optimizer,
                                       train_loss=train_loss, valid_loss=val_loss,
                                       train_score=train_acc, valid_score=val_acc,
                                       batch_size=args.batch_size, epoch=t, path=model_path)
            best_acc = val_acc

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
            df.to_csv(history_path)
            break

    print("Done!")
    # Save history
    df = pandas.DataFrame(history)
    df.to_csv(history_path)


if __name__ == '__main__':
    main()
