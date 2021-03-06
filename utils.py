import argparse
import datasets
import pprint
import pytorch_lightning as pl
import os


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
    parser.add_argument('--model_size', type=str, default='small', help='Model size.', 
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--lr', type=float, default=3e-4, help='The learning rate.')
    parser.add_argument('--wd', type=float, default=1e-3, help='The weight decay.')
    parser.add_argument('--exp_id', type=str, default='', help='The experiment id.')
    parser.add_argument('--logger', type=str, default='wandb', help='The logger to use.', 
                        choices=['wandb', 'tensorboard'])
    args = parser.parse_args()

    if verbose:
        args_dict = vars(args)
        args_dict = {k: v for k, v in sorted(list(args_dict.items()))}
        pprint.pprint(args_dict)
    return args

def get_logger(args):
    """
    Logger for the PyTorchLightning Trainer.
    """
    logger_kind = 'tensorboard' if 'logger' not in args.__dict__ else args.logger
    if logger_kind == 'tensorboard':
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'tmp'),
            name=args.dataset,
        )

    elif logger_kind == 'wandb':
        task_str = [args.task]
        name = [
            str(args.exp_id), args.dataset, '-'.join(task_str),
            '-'.join([str(args.model_size), str(args.patch_size)])
        ]
        logger = pl.loggers.WandbLogger(
            save_dir=os.path.join(os.getcwd(), 'tmp'),
            name='/'.join(name),
            project='glom',
        )

    else:
        raise Exception(f'Error. Logger "{lokker_kind}" is not supported.')
    return logger
