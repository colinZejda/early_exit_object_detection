import os
import numpy as np
import torch 
from torchvision import datasets, transforms
# from torch.utils.data.distributed import DistributedSampler  # not using distributed system here
from torch.utils.data.sampler import SubsetRandomSampler

def data_transforms(data_transform_type):
    """get transform of dataset"""
    if data_transform_type in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            #lighting_param = 0.1
        elif data_transform_type == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            #lighting_param = 0.1
        elif data_transform_type == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            #lighting_param = 0.1
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transforms = None
        val_transforms = None

    return train_transforms, val_transforms

def dataset(test_only, train_transforms, val_transforms, dataset_dir):
    """get dataset for classification"""
    if not test_only:
        train_set = datasets.ImageFolder(
            os.path.join(dataset_dir, 'train'),
            transform=train_transforms)
    else:
        train_set = None
    val_set = datasets.ImageFolder(
        os.path.join(dataset_dir, 'val'),
        transform=val_transforms)
    return train_set, val_set


def data_loader(test_only, train_set, val_set, batch_size, shuffle=True, valid_size=0.1):

    # Don't worry about this part its for multi gpu work
    # infer batch size
    # if getattr(FLAGS, 'batch_size', False):
    #     if getattr(FLAGS, 'batch_size_per_gpu', False):
    #         assert FLAGS.batch_size == (
    #             FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job)
    #     else:
    #         assert FLAGS.batch_size % FLAGS.num_gpus_per_job == 0
    #         FLAGS.batch_size_per_gpu = (
    #             FLAGS.batch_size // FLAGS.num_gpus_per_job)
    # elif getattr(FLAGS, 'batch_size_per_gpu', False):
    #     FLAGS.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus_per_job
    # else:
    #     raise ValueError('batch size (per gpu) is not defined')
    # batch_size = int(FLAGS.batch_size/get_world_size())

    # if getattr(FLAGS, 'distributed', False):
    #     if FLAGS.test_only:
    #         train_sampler = None
    #     else:
    #         train_sampler = DistributedSampler(train_set)
    #     val_sampler = DistributedSampler(val_set)
    # else:
    #     train_sampler = None
    #     val_sampler = None

    # TRAIN + VAL SAMPLERS
    train_sampler = None
    val_sampler = None

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    #train_idx, valid_idx = indices[split:], indices[:split]
    #train_sampler = SubsetRandomSampler(train_idx)
    #val_sampler = SubsetRandomSampler(valid_idx)

    # CREATE DATA LOADERS 
    if not test_only:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            num_workers=1,
            drop_last=True)
    else:
        train_loader = None

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=1,
        drop_last=True)

    return train_loader, val_loader