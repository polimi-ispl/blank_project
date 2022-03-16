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


def test_loop(dataloader, model, loss_fn, platform):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # Move data on GPU
            X = X.to(platform)
            y = y.to(platform)
            # Prediction on the validation set
            pred = model(X)
            loss += loss_fn(pred, y.type(torch.long)).item()
            correct += (pred.argmax(1) == y).sum().item()

    loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")
    return correct, loss


def main():
    parser = ArgumentParser(description="An example of training a CIFAR classifier in pytorch")
    parser.add_argument("--runpath", type=str, required=False,
                        default="./data/trained_models/pytorch",
                        help="Results directory to be loaded")
    parser.add_argument("--gpu", type=int, required=False, default=None,
                        help="Index of GPU to use (None for CPU, -1 for least used GPU)")
    args = parser.parse_args()

    # load args from runpath, to check the training parameters
    trainargs = src.read_args(filename=os.path.join(args.runpath, "args.txt"))
    
    model_path = os.path.join(args.runpath, "best_model.pth")
    history_path = os.path.join(args.runpath, "history.csv")
    
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

    # Load test set, initialize Dataloaders
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    # Initialize model
    model = arch.model_torch()
    model = src.torch_utils.load_weights(model, model_path)

    # Move the model on gpu either with
    model = model.to(platform)
    
    # define the optimization parameters
    loss_fn = nn.CrossEntropyLoss()
    
    accuracy, test_loss = test_loop(loader, model, loss_fn, platform)
    
    print("Done")


if __name__ == '__main__':
    main()
