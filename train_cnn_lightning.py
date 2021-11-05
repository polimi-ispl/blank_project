"""
An example of training script that implements Pytorch-Lightning
@Author: Francesco Picetti
"""
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'pytorch_lightning'])
    import pytorch_lightning as pl
    

class CNN(pl.LightningModule):
    def __init__(self, n_classes=10):
        """A standard convolutional classifier"""
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3)  # 30x30
        self.pool1 = torch.nn.MaxPool2d(2, 2)   # 15x15
        self.conv2 = torch.nn.Conv2d(32, 64, 3)  # 13x13
        self.pool2 = torch.nn.MaxPool2d(2, 2)    # 6x6, no padding
        self.conv3 = torch.nn.Conv2d(64, 64, 3)  # 4x4
        self.dense1 = torch.nn.Linear(4*4*64, 64)
        self.dense2 = torch.nn.Linear(64, n_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer"   : optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True),
                "monitor"  : "val_loss",
            },
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss


def main():
    parser = ArgumentParser(description="An example of parsing arguments")
    
    parser.add_argument("--outpath", type=str, required=False,
                        default="results",
                        help="Results directory")
    parser.add_argument("--num_gpus", type=int, required=False, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, required=False, default=32,
                        choices=[16, 32, 64],
                        help="Batch size")
    parser.add_argument("--epochs", type=int, required=False, default=100,
                        help="Max iterations number")
    args = parser.parse_args()
    
    # Output path

    # Transform to tensor and normalize to [0, 1]
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load training and validation set, initialize Dataloaders
    trainset = CIFAR10(root='./data', train=True, download=True, transform=trans)
    train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valset = CIFAR10(root='./data', train=False, download=True, transform=trans)
    val_dataloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # initialize a model
    cnn = CNN()
    
    # define callbacks
    early_stopping = pl.callbacks.EarlyStopping('val_loss', mode="min", patience=10)
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=args.outpath)
    
    # initialize a trainer
    trainer = pl.Trainer(gpus=args.num_gpus,  # how many GPUs to use...
                         auto_select_gpus=True,  # ... only if they are available
                         deterministic=True,  # enables reproducibility
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=20,
                         callbacks=[early_stopping, checkpoint])
    
    # train the model
    trainer.fit(cnn, train_dataloader, val_dataloader)
    
    print("Done!")


if __name__ == '__main__':
    main()

