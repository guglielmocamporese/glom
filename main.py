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
        model = GlomClassification(img_size=ds_info['img_size'], patch_size=args.patch_size)
    elif args.task == 'reconstruction': 
        model = GlomReconstruction(img_size=ds_info['img_size'], patch_size=args.patch_size)
    else:
        raise Exception(f'Error. Task "{args.task}" is not supported.')

    # Create trainer
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus)

    # Fit
    trainer.fit(model, train_dataloaders=dl_dict['train'], val_dataloaders=dl_dict['val'])


if __name__ == '__main__':

    # Retrieve input args
    args = utils.parse_args(sys.argv[1:])

    # Run main
    main(args)
