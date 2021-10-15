"""
Example script for using TensorBoard
https://tensorboardx.readthedocs.io/en/latest/tutorial.html#install

@Author: Francesco Picetti
@Author: Nicolò Bonettini
"""
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


def main():
    # initialize tensorboard
    # see https://tensorboardx.readthedocs.io/en/latest/tutorial.html#create-a-summary-writer
    tb = SummaryWriter(logdir="./logs")
    
    for epoch in range(100):
        x = np.random.randn(3, 3)
        loss = np.linalg.norm(x)
        
        # add a scalar value to be plot as history on TB
        tb.add_scalar("train/loss", loss, epoch)
        
        # add a image to be shown on TB
        fig, ax = plt.subplots(1, 1)
        ax.set_title("Training vector")
        ax.imshow(x, cmap="gray")

        tb.add_figure('train/vector', fig, epoch)
        
        # force TB write on disk
        tb.flush()
        
    tb.close()
    
    
if __name__ == "__main__":
    main()
    