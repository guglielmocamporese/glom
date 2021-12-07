import sys
import pytorch_lightning as pl

from glom import GlomReconstruction, GlomClassification
from datasets import get_dataloaders
import utils


def main(args):

    # Dataset and dataloader
    dl_dict, ds_info = get_dataloaders(args)

    # Model and trainer
    if args.task == 'classification': 
        model = GlomClassification(args, img_size=ds_info['img_size'], patch_size=args.patch_size, 
                                   num_classes=ds_info['num_classes'], in_chans=ds_info['in_chans'])
    elif args.task == 'reconstruction': 
        model = GlomReconstruction(args, img_size=ds_info['img_size'], patch_size=args.patch_size, 
                                   in_chans=ds_info['in_chans'])
    else:
        raise Exception(f'Error. Task "{args.task}" is not supported.')

    # Logger
    logger = utils.get_logger(args)

    # Create trainer
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, logger=logger)

    # Fit
    trainer.fit(model, train_dataloaders=dl_dict['train'], val_dataloaders=dl_dict['val'])


if __name__ == '__main__':

    # Retrieve input args
    args = utils.parse_args(sys.argv[1:])

    # Run main
    main(args)
