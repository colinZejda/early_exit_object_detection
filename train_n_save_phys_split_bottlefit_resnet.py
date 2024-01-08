"""
Currently on IAS lab 2019 server, Ian's account
Goal: train ResNet50 with physical split (bottlefit) on imageNet data

path to FULL imagenet train data: '/home/ian/dataset/ImageNet/train/'
path to FULL imagenet val data:   '/home/ian/dataset/ImageNet/val/'
"""

import gc 
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models

from custom_wrapper import ResNet50Custom
from proper_imagenet_dataloaders import data_transforms, dataset, data_loader
from resnet50 import ResNet, ResidualBlock50
from split_resnet50 import ResNetHead, ResNetTail


# train student head + encoder/decoder, freeze tail
def first_round_training(device, train_loader, valid_loader):
    # setting hyperparams
    num_epochs = 100
    batch_size = 200
    learning_rate = 1e-3

    # set up teacher and student models (resnet50)
    # teacher_path = '/home/ian/colin_early_exit_dec2023/early_exit_object_detection/original_resnet50_epoch50.pth'          # model+architecture, no need to instantiate the class ResNet
    # teacher_model = ResNet(ResidualBlock50, [3, 4, 6, 3]).to(device)
    # teacher = teacher_model.load_state_dict(torch.load(teacher_path))
    teacher = models.resnet50(pretrained=True).to(device)
    teacher_layer_outputs = ResNet50Custom().to(device)

    student_head = ResNetHead().to(device)
    student_tail = ResNetTail(ResidualBlock50, [3, 4, 6, 3]).to(device)

    # copy weights over (these will be frozen in student)
    x = teacher.layer2
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
    best_val_accuracy= 0                              # for saving the best model found
    best_student_head, best_student_tail = None, None
    for epoch in range(num_epochs):
        for i, (images, labels, _) in tqdm(enumerate(train_loader)):

            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            student_out = student_head(images)
            student_out = student_tail(student_out)
            with torch.no_grad():
                teacher_out = teacher(images)
                T_l1_out, T_l2_out, T_l3_out = teacher_layer_outputs(images)

            # calculate loss
            
            loss = loss_fn(student_tail.x, T_l1_out)                  # primary loss (remember, loss of outputs), x = student out of decoder, l1 = teacher out of layer1 
            loss =  loss_fn(student_tail.l2_out, T_l2_out)            # l2 and l3 just help
            loss += loss_fn(student_tail.l3_out, T_l3_out)
            loss_acumm = loss.item()

            # backprop
            loss.backward()
            student_optimizer.step()
            student_optimizer.zero_grad()

            del images, labels, student_out, teacher_out
            torch.cuda.empty_cache()
            gc.collect()

        # Save model state dict (weights + architecture)
        if (epoch+1) % 20 == 0 or epoch+1 == num_epochs:               # save 2 models: halfway and end
            path_to_save_head = f"1st_round_student_HEAD_resnet50_epoch{epoch+1}.pth"
            path_to_save_tail = f"1st_round_student_TAIL_resnet50_epoch{epoch+1}.pth"
            torch.save(best_student_head.state_dict(), path_to_save_head)               # to load: model.load_state_dict(torch.load(path_to_save)), after creating instance of object, model
            torch.save(best_student_tail.state_dict(), path_to_save_tail)
            print(f"Model on epoch {epoch+1} saved!!!!\n")

        # Train loss
        print('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, loss_acumm/total_step))

        # Validation accuracy (per epoch)
        total_val_imgs = len(valid_loader) * batch_size
        with torch.no_grad():
            num_correct = 0
            for images, labels, _ in tqdm(valid_loader):            # loop over all batches
                images = images.to(device)
                labels = labels.to(device)

                student_output1 = student_head(images)           # forward pass
                student_output2 = student_tail(student_output1)

                for i in range(student_output2.shape[0]):        # loop within a batch
                    predicted = torch.argmax(student_output2[i])
                    if predicted == labels[i].item():
                        num_correct += 1
                del images, labels, student_output2

            val_acc = 100*(num_correct/total_val_imgs)
            if val_acc > best_val_accuracy:
                best_student_head = student_head
                best_student_tail = student_tail
            print('Val accuracy on {} validation images: {:.2f} %'.format(total_val_imgs, val_acc))

# reverse the freeze, only train tail
def second_round_training(device, train_loader, valid_loader):

    # setting hyperparams
    num_epochs = 100
    batch_size = 200
    learning_rate = 1e-3

    # path to head+tail
    first_stage_done_head_path = "1st_round_student_HEAD_resnet50_epoch100.pth"
    first_stage_done_tail_path = "1st_round_student_TAIL_resnet50_epoch100.pth"

    # instantiate student head + tail
    student_head = ResNetHead().to(device)
    student_tail = ResNetTail(ResidualBlock50, [3, 4, 6, 3]).to(device)

    # load state dicts
    student_head.load_state_dict(torch.load(first_stage_done_head_path))
    student_tail.load_state_dict(torch.load(first_stage_done_tail_path))
    student_head.to(device)
    student_tail.to(device)
    
    # params for optimizer when training (just tail, head + decoder frozen)
    student_params = list(student_tail.layer2.parameters()) + list(student_tail.layer3.parameters()) + list(student_tail.fc.parameters())

    # Loss func and optimizer
    loss_fn = nn.CrossEntropyLoss()             # 2nd stage of training uses CCE (classification, goes thru to fc layer in tail)
    student_optimizer = torch.optim.SGD(student_params, lr=learning_rate, weight_decay = 1e-3, momentum = 1e-3)

    # TRAIN LOOP
    total_step = len(train_loader)
    loss_acumm = 0
    best_val_accuracy= 0                              # for saving the best model found
    best_student_head, best_student_tail = None, None

    for epoch in range(num_epochs):
        for i, (images, labels, _) in tqdm(enumerate(train_loader)):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            student_out = student_head(images)
            student_out = student_tail(student_out)

            # calculate loss
            loss = loss_fn(student_out, labels)       # CCE loss
            loss_acumm += loss.item()

            # backprop
            loss.backward()
            student_optimizer.step()
            student_optimizer.zero_grad()

            del images, labels, student_out
            torch.cuda.empty_cache()
            gc.collect()

        # Save model state dict (weights + architecture)
        if (epoch+1) % 20 == 0 or epoch+1 == num_epochs:               # save best models as we train
            path_to_save_head = f"best_acc_student_HEAD_resnet50_epoch{epoch+1}.pth"
            path_to_save_tail = f"best_acc_student_TAIL_resnet50_epoch{epoch+1}.pth"
            torch.save(best_student_head.state_dict(), path_to_save_head)               # to load: model.load_state_dict(torch.load(path_to_save)), after creating instance of object, model
            torch.save(best_student_tail.state_dict(), path_to_save_tail)
            print(f"Best model on epoch {epoch+1} saved!!!!\n")

        # Train loss
        print('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, loss_acumm/total_step))

        # Validation accuracy (per epoch)
        total_val_imgs = len(valid_loader) * batch_size
        with torch.no_grad():
            num_correct = 0
            for images, labels, _ in tqdm(valid_loader):            # loop over all batches
                images = images.to(device)
                labels = labels.to(device)

                student_output1 = student_head(images)           # forward pass
                student_output2 = student_tail(student_output1)

                for i in range(student_output2.shape[0]):        # loop within a batch
                    predicted = torch.argmax(student_output2[i])
                    if predicted == labels[i].item():
                        num_correct += 1
                del images, labels, student_output2
            val_acc = 100*(num_correct/total_val_imgs)
            if val_acc > best_val_accuracy:
                best_student_head = student_head
                best_student_tail = student_tail
            print('Val accuracy on {} validation images: {:.2f} %'.format(total_val_imgs, val_acc))


def main():
    # create data loaders
    path_to_imagenet = "/home/ian/dataset/1Per"                              # using 1Per for the moment for testing 
    train_transforms, val_transforms = data_transforms('imagenet1k_basic')   # unsure about 1k basic?
    train_set, val_set = dataset(False, train_transforms, val_transforms, path_to_imagenet)
    train_loader, val_loader = data_loader(False, train_set, val_set, 200)   # 200 batch size

    # setting CUDA, allowed to use GPU0
    gpu_id = 2
    device = f'cuda:{gpu_id}'

    # perform training
    first_round_training(device, train_loader, val_loader)      # train head + encoder/decoder, freeze tail
    second_round_training(device, train_loader, val_loader)     # reverse the freeze, only train tail
    
    print("Fully trained Bottlefit ResNet50 saved as:", "")     # to be filled

if __name__ == '__main__':
    main()