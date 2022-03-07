"""
Check GPU status

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
Francesco Picetti
Nicolò Bonettini
Francesco Maffezzoli
Edoardo Daniele Cannas
"""
import numpy as np
import torch


# Classes and helpers functions #

class DummyNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(DummyNeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def main():
    # Check if GPUs exist
    print(f'Torch CUDA is available: {torch.cuda.is_available()}')
    print('Num GPUs Available: ', torch.cuda.device_count())
    print(f'Names of the available GPUs: ')
    for gpu_idx in range(torch.cuda.device_count()):
        print(f'{gpu_idx}:  {torch.cuda.get_device_name(gpu_idx)}')

    # Run a CNN
    print('Define a CNN model:')
    model = DummyNeuralNetwork()
    print(model)
    print('Moving the model to GPU...')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # we'll use the first GPU available
    model.to(device)
    print(f'Is model on GPU? {next(model.parameters()).is_cuda}')  # returns a boolean

    print('Run a quick inference:')
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = torch.nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    print('Test finished')


if __name__ == '__main__':
    main()
