"""
Example script for passing user-defined parameters to a script.

@Author: Francesco Picetti
"""

# method 1: define a "params.py" file and import the variables
# pros: easy to understand, easy to use
# cons: not so flexible, you need to create a params file for every combination of parameters
from params import trained_models_root, model_name_keras, input_shape, batch_size, epochs

print(trained_models_root)
print(model_name_keras)
print(input_shape)
print(batch_size)
print(epochs)

input_shape = (32, 32, 3)

# method 2: use ArgumentParser (https://docs.python.org/3/library/argparse.html)

from argparse import ArgumentParser
import os

if __name__ == "__main__":
    
    parser = ArgumentParser(description="An example of parsing arguments")
    
    parser.add_argument("--trained_models_root", type=str, required=False,
                        default=os.path.join("data", 'trained_models'),
                        help="Name of the model to be loaded")
    parser.add_argument("--model_name_keras", type=str, required=False, default="model_keras",
                        help="Name of the model to be loaded")
    parser.add_argument("--input_shape", type=int, nargs="+", required=False,
                        help="input shape")
    parser.add_argument("--batch_size", type=int, required=False, default=32,
                        choices=[16, 32, 64],
                        help="Batch size")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Max iterations number")
    
    # then you can parse the argument from the bash command "python pass_parameters_to_script.py --epochs 100"
    args = parser.parse_args()

    # you can access the arguments as attributes of args
    print(args.trained_models_root)
    print(args.model_name_keras)
    print(args.input_shape)
    print(args.batch_size)
    print(args.epochs)
    
    # note that args.input_shape is None; if you want to have a default tuple, you have to define the following:
    if args.input_shape is None:
        args.input_shape = (32,32,3)
        
    # otherwise, you can pass arguments with "python pass_parameters_to_script.py --epochs 100 --input_shape 32 32 3"
    # but watch out! this will be a list, not a tuple! So a better procedure is:
    if args.input_shape is None:
        args.input_shape = [32, 32, 3]
    args.input_shape = tuple(args.input_shape)
    
    assert isinstance(args.input_shape, tuple)
    