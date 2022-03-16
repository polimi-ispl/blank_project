"""
An example of test script that implements Pytorch-Lightning
@Author: Francesco Picetti
"""
from argparse import ArgumentParser
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import src

from train_cnn_lightning import CNN


try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install Pytorch Lightning with: `pip install pytorch_lightning`")


class CNN_with_test(CNN):
    def __init__(self, *args, **kwargs):
        super(CNN_with_test, self).__init__(*args, **kwargs)
    
    # add here the test routine (a cleaner way is to define the whole model in another file,
    # to be loaded in both the training and testing scripts
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log('test_loss', test_loss)
        # here you can compute (and log) as many metrics as you want
        return test_loss


def main():
    parser = ArgumentParser(description="Test a CIFAR classifier based on PyTorch Lightning")
    
    parser.add_argument("--runpath", type=str, required=False,
                        default="./data/trained_models/lightning",
                        help="Results directory to be loaded")
    parser.add_argument("--num_gpus", type=int, required=False, default=1,
                        help="Number of GPUs to use")
    args = parser.parse_args()
    
    # load args from runpath, to check the training parameters
    trainargs = src.read_args(filename=os.path.join(args.runpath, "args.txt"))
    
    # Transform to tensor and normalize to [0, 1]
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load test set, initialize Dataloaders
    testset = CIFAR10(root='./data', train=False, download=True, transform=trans)
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    # initialize a trainer
    trainer = pl.Trainer(gpus=args.num_gpus,  # how many GPUs to use...
                         auto_select_gpus=True if args.num_gpus != 0 else False,  # ... only if they are available
                         )
    
    # test the model
    trainer.test(model=CNN_with_test(),
                 dataloaders=dataloader,
                 ckpt_path=os.path.join(args.runpath, "best_model.ckpt"))
    
    # you can access the loss and other metrics in the trainer attributes
    print("Test Loss = %.2e" % trainer.callback_metrics["test_loss"])
    
    print("Done!")


if __name__ == '__main__':
    main()

