"""
Currently on IAS lab 2019 server, ian's account
Goal: train ResNet50 with physical split (bottlefit) on imageNet data

path to imagenet train data: '/home/ian/dataset/1Perhold_out1/train/'
path to imagenet val data:   '/home/ian/dataset/1Perhold_out1/val/'
"""

import os
import gc 
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms

from og_resnet_backbone import ResidualBlock50, ResNet
from phys_split_resnet_backbone import encoder, decoder, ResNetHead, ResNetTail

# DATA LOADER FUNC, uses Imagefolder from pytorch (for ImageNet dataset specifically)
def data_loader(train_data_dir, validation_data_dir, batch_size, random_seed=42, valid_size=0.1, shuffle=True, test=False):

    # normalize data (helps convergence during training)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            normalize
    ])

    transform_valid = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
    ])

    # load dataset
    train_dataset = ImageFolder(root = train_data_dir, transform = transform_train)
    valid_dataset = ImageFolder(root = validation_data_dir, transform = transform_valid)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1)

    return (train_loader, valid_loader)


# train student head + encoder/decoder, freeze tail
def first_round_training(device, train_loader, valid_loader):
    # setting hyperparams
    num_classes = 100
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3

    # set up teacher and student models (resnet50)
    teacher_path = '/home/ian/'          # model+architecture, no need to instantiate the class ResNet
    teacher = torch.load(teacher_path).to(device)
    student_head = ResNetHead(ResidualBlock50, [3, 4, 6, 3], num_classes=num_classes).to(device)
    student_tail = ResNetTail(ResidualBlock50, [3, 4, 6, 3], num_classes=num_classes).to(device)

    # copy weights over (these will be frozen in student)
    student_tail.layer2 = teacher.layer2
    student_tail.layer3 = teacher.layer3
    student_tail.fc = teacher.fc

    # params for optimizer when training (just head + decoder, tail frozen)
    student_params = list(student_head.parameters()) + list(student_tail.decoder.parameters())

    # Loss func (criterion) and optimizers
    loss_fn = nn.MSELoss()
    student_optimizer = torch.optim.SGD(student_params, lr=learning_rate, weight_decay=1e-3, momentum=1e-3)

    # TRAIN LOOP
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in tqdm(enumerate(train_loader)):

            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            student_out = student_head(images)
            student_out = student_tail(student_out)
            with torch.no_grad():
                teacher_out = teacher(images)

            # calculate loss
            loss = loss_fn(student_tail.x, teacher.l1_out)                  # primary loss (remember, loss of outputs)
            loss =  loss_fn(student_tail.l2_out, teacher.l2_out)            # l2 and l3 just help
            loss += loss_fn(student_tail.l3_out, teacher.l3_out)
            loss_acumm = loss.item()

            # backprop
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

            del images, labels, student_out, teacher_out
            torch.cuda.empty_cache()
            gc.collect()

        # Save model state dict (weights + architecture)
        if epoch+1 == num_epochs//2 or epoch+1 == num_epochs:               # save 2 models: halfway and end
            path_to_save_head = f"student_HEAD_resnet50_epoch{epoch+1}.pth"
            path_to_save_tail = f"student_TAIL_resnet50_epoch{epoch+1}.pth"
            torch.save(student_head.state_dict(), path_to_save_head)               # to load: model.load_state_dict(torch.load(path_to_save)), after creating instance of object, model
            torch.save(student_tail.state_dict(), path_to_save_tail)
            print(f"Model on epoch {epoch+1} saved!!!!\n")

        # Train loss
        print('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, loss_acumm/total_step))

        # Validation accuracy (per epoch)
        total_val_imgs = len(valid_loader) * batch_size
        with torch.no_grad():
            num_correct = 0
            for images, labels in tqdm(valid_loader):            # loop over all batches
                images = images.to(device)
                labels = labels.to(device)

                student_output1 = student_head(images)           # forward pass
                student_output2 = student_tail(student_output1)

                for i in range(student_output2.shape[0]):        # loop within a batch
                    predicted = torch.argmax(student_output2[i])
                    if predicted == labels[i].item():
                        num_correct += 1
                del images, labels, student_output2

            print('Val accuracy on {} validation images: {:.2f} %'.format(total_val_imgs, 100*(num_correct/total_val_imgs)))


# reverse the freeze, only train tail
def second_round_training():
    pass

def main():
    # create data loaders
    path_to_imagenet = ""   # in progress, need ian's help
    train_loader, valid_loader = data_loader(train_data_dir=path_to_imagenet+'/train',
				                                  validation_data_dir=path_to_imagenet+'/val',
				                                  batch_size=64)
    # setting CUDA, allowed to use GPU0
    gpu_id = 0
    device = f'cuda:{gpu_id}'

    # perform training
    first_round_training(device, train_loader, valid_loader)  # train head + encoder/decoder, freeze tail
    second_round_training()                     # reverse the freeze, only train tail
    
    print("Fully trained Bottlefit ResNet50 saved as:", "")

if __name__ == '__main__':
    main()