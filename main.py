from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from glom import Glom


def main():

    # Dataset and dataloader
    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform_train = T.Compose([
        T.RandomResizedCrop(32),
        T.ToTensor(),
        normalize,
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    ds_train = CIFAR10('./data', train=True, transform=transform_train, download=True)
    ds_val = CIFAR10('./data', train=False, transform=transform_val, download=True)
    dl_train = DataLoader(ds_train, shuffle=True, pin_memory=True, num_workers=12, batch_size=64)
    dl_val = DataLoader(ds_val, pin_memory=True, num_workers=12, batch_size=64)

    # Model and trainer
    model = Glom(img_size=32, patch_size=4)
    trainer = pl.Trainer(max_epochs=5, gpus=1)

    # Fit
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)


if __name__ == '__main__':
    main()
