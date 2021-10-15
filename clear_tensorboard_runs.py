"""
@Author: Nicolò Bonettini
"""
import sys
sys.path.insert(0, '..')
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from tensorboard.backend.event_processing import event_accumulator as ea


def main():
    """
    This script scans the Tensorboard run folder for aborted runs (less than `min` logging events) and remove them from
    disk. After finding the runs, the script prints a regex to paste into Tensorboard to perform a last check, and waits
    for a key pressed to confirm the deletion, or Ctrl-C to abort.
    Keep in mind that it's better to stop Tensorboard just before confirming the deletion, to avoid file locks.
    You can use `-n` param to perform a dry run without actually deleting folders on disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_root', type=str, required=True)
    parser.add_argument('--min', help='Minimum number of logging events to consider', default=100, type=int)
    parser.add_argument('-n', '--dry_run', help='Perform a dry-run instead of actually removing files',
                        action='store_true')

    args = parser.parse_args()
    min_event = args.min
    dry_run = args.dry_run

    all_runs = os.listdir(args.run_root)

    to_remove = []
    for run_folder in tqdm(all_runs):
        file_list = os.listdir(os.path.join(args.run_root, run_folder))

        # if we have a single 0-bytes file in the run_folder, we remove it directly
        if len(file_list) == 1:
            if os.path.getsize(os.path.join(args.run_root, run_folder, file_list[0])) == 0:
                to_remove.append(run_folder)
                continue

        # if we don't have weights stored, we remove it directly:
        if not np.any(['.pth' in x for x in file_list]):
            to_remove.append(run_folder)
            continue

        acc = ea.EventAccumulator(os.path.join(args.run_root, run_folder))
        acc.Reload()

        try:
            scalars_event = acc.Scalars('lr')
        except KeyError:
            continue

        if len(scalars_event) < min_event:
            to_remove.append(run_folder)

    [print(f'Removing {x}') for x in to_remove]

    regexp = f"({'|'.join([x.rsplit('-', 1)[-1] for x in to_remove])})"
    print(f'\nYou can check this regexp in Tensorboard: \n{regexp}\n')
    input('Press a key to continue, or Ctrl-C to terminate')

    if not dry_run:
        [shutil.rmtree(os.path.join(args.run_root, x), ignore_errors=True) for x in to_remove]

    print(f'Removed {len(to_remove)} folders out of {len(all_runs)}')
    if dry_run:
        print('I\'m joking, this was just a dry run. Remove the --dry_run argument to actually delete files.')

    return 0


if __name__ == '__main__':
    main()
