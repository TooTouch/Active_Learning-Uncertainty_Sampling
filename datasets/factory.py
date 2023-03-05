import os
from torch.utils.data import DataLoader
from .augmentation import train_augmentation, test_augmentation


def create_dataset(datadir: str, dataname: str):
    trainset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir,dataname), 
        train     = True, 
        download  = True, 
        transform = train_augmentation()
    )
    testset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir,dataname), 
        train     = False, 
        download  = True, 
        transform = test_augmentation()
    )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers
    )
