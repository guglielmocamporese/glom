import os
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader

__available__ = ['cifar10', 'cifar100', 'imagenet']

NORMALIZATION = {
    'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)],
    'cifar100': [(0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2761)],
    'tiny_imagenet': [(0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)],
    'caltech256': [(0.5415, 0.5187, 0.4880), (0.3081, 0.3040, 0.3169)],
    'flower102': [(0.5115, 0.4159, 0.3407), (0.2957, 0.2493, 0.2889)],
    'oxford_pet': [(0.4830, 0.4448, 0.3956), (0.2591, 0.2531, 0.2596)],
    'cars': [(0.4460, 0.4311, 0.4318), (0.2903, 0.2884, 0.2955)],
    'svhn': [(0.4378, 0.4439, 0.4729), (0.1981, 0.2011, 0.1970)],
}

def get_datasets(args):
    """
    Return datasets.
    """
    ds_kwargs = {
        'root': os.path.join(args.data_path, args.dataset),
        'download': True,
    }
    mu, std = NORMALIZATION[args.dataset]
    if args.dataset == 'mnist':
        ds_train = MNIST(train=True, transforms=T.ToTensor(), **ds_kwargs)
        ds_val = MNIST(train=False, transforms=T.ToTensor(), **ds_kwargs)

    elif args.dataset == 'cifar10':
        normalize = T.Normalize(mean=mu, std=std)
        train_transform = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        ds_train = CIFAR10(train=True, transform=train_transform, **ds_kwargs)
        ds_val = CIFAR10(train=False, transform=val_transform, **ds_kwargs)
        ds_info = {
            'in_channels': 3,
            'img_size': 32,
        }

    elif args.dataset == 'cifar100':
        normalize = T.Normalize(mean=mu, std=std)
        train_transform = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        ds_train = CIFAR100(train=True, transform=train_transform, **ds_kwargs)
        ds_val = CIFAR100(train=False, transform=val_transform, **ds_kwargs)
        ds_info = {
            'in_channels': 3,
            'img_size': 32,
        }

    elif args.dataset == 'imagenet':
        normalize = T.Normalize(mean=mu, std=std)
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ])
        ds_train = ImageFolder(args.data_path, transform=train_transform)
        ds_val = ImageFolder(args.data_path, transform=val_transform)
        ds_info = {
            'in_channels': 3,
            'img_size': 224,
        }

    else:
        raise Exception(f'Error. Dataset "{args.dataset}" not supported.')
    ds_dict = {
        'train': ds_train,
        'val': ds_val,
    }
    return ds_dict, ds_info

def get_dataloaders(args):
    """
    Return dataloaders.
    """
    ds_dict, ds_info = get_datasets(args)
    dl_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    dl_dict = {
        'train': DataLoader(ds_dict['train'], shuffle=True, **dl_kwargs),
        'val': DataLoader(ds_dict['val'], **dl_kwargs),
    }
    return dl_dict, ds_info
