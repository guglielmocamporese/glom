import argparse
import datasets
import pprint


def parse_args(stdin, verbose=True):
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(stdin)
    parser.add_argument('--dataset', type=str, choices=datasets.__available__, help='Dataset to use.')
    parser.add_argument('--data_path', type=str, default='./data', help='Dataset root path.')

    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers to use for the dataloader.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for the training process.')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size.')
    parser.add_argument('--task', type=str, default='classification', help='The task to solve with the GLOM model.', 
                        choices=['classification', 'reconstruction'])
    args = parser.parse_args()

    if verbose:
        args_dict = vars(args)
        args_dict = {k: v for k, v in sorted(list(args_dict.items()))}
        pprint.pprint(args_dict)
    return args
