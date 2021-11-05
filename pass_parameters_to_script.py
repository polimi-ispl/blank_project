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

# method 2: use ArgumentParser (https://docs.python.org/3/library/argparse.html
from argparse import ArgumentParser
import os


if __name__ == "__main__":
    
    parser = ArgumentParser(description="An example of argument parsing")
    
    parser.add_argument("--outpath", type=str, required=False,
                        default="results",
                        help="Run folder to store results")
    parser.add_argument("--input_shape", type=int, nargs="+", required=False,
                        help="input shape")
    parser.add_argument("--batch_size", type=int, required=False, default=32,
                        choices=[16, 32, 64],
                        help="Batch size")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Max iterations number")
    parser.add_argument("--verbose", required=False, action="store_true", default=False,
                        help="Print the arguments")
    
    # then you can parse the argument from the bash command "python pass_parameters_to_script.py --epochs 100"
    args = parser.parse_args()
    
    # you can access the arguments as attributes of args
    if args.verbose:
        print(args.outpath)
        print(args.input_shape)
        print(args.batch_size)
        print(args.epochs)
    
    # note that args.input_shape is None; if you want to have a default tuple, you have to define the following:
    if args.input_shape is None:
        args.input_shape = (32, 32, 3)
    
    # otherwise, you can pass arguments with "python pass_parameters_to_script.py --epochs 100 --input_shape 32 32 3"
    # watch out! this will be a list, not a tuple! So a better procedure is:
    if args.input_shape is None:
        args.input_shape = [32, 32, 3]
    args.input_shape = tuple(args.input_shape)
    
    assert isinstance(args.input_shape, tuple)
    
    # now we can save the arguments to a txt file in the run folder, in order to keep memory of the hyper-parameters that
    # have generated that specific run.
    from src import write_args, read_args
    os.makedirs(args.outpath, exist_ok=True)
    write_args(filename=os.path.join(args.outpath, "args.txt"), args=args)
    
    # if you need to load the args from such a file, simply use read_args:
    args_from_disk = read_args(filename=os.path.join(args.outpath, "args.txt"))
