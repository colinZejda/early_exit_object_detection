import gc 
from tqdm import tqdm

import torch
import torch.nn as nn

from proper_imagenet_dataloaders import data_transforms, dataset, data_loader
from resnet50 import ResidualBlock50, ResNet

def train_n_save(device, train_loader, val_loader):
    # Setting Hyperparameters
    num_classes = 1000           # for full imagenet class (for 1% it's 100 classes)
    num_epochs = 100
    batch_size = 128
    learning_rate = 1e-3

    # instantiate model
    model = ResNet(ResidualBlock50, [3, 4, 6, 3], num_classes=num_classes).to(device)

    # loss and optimzer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-3, momentum=0.9)

    # TRAIN LOOP
    total_step = len(train_loader)
    loss_acumm = 0
    best_val_accuracy, best_model_found = 0, None                # for saving the best model found
    for epoch in range(num_epochs):
        for i, (images, labels, _) in tqdm(enumerate(train_loader)):

            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.long().to(device)

            # Forward pass
            y_hat = model(images)
            loss = criterion(y_hat, labels)
            loss_acumm += loss.item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del images, labels, y_hat
            torch.cuda.empty_cache()
            gc.collect()

        # Save model state dict (weights + architecture)
        if (epoch+1) % 10 == 0 or epoch+1 == num_epochs:
            path_to_save_resnet = f"best_acc_original_resnet50_epoch{epoch+1}.pth"
            torch.save(best_model_found.state_dict(), path_to_save_resnet)               # to load: model.load_state_dict(torch.load(path_to_save)), after creating instance of object, model
            print(f"Model on epoch {epoch+1} saved!!!!\n")

        # Train loss
        print('Epoch [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, loss_acumm/total_step))

        # Validation accuracy (per epoch)
        total_val_imgs = len(val_loader) * batch_size
        with torch.no_grad():
            num_correct = 0
            for images, labels, _ in tqdm(val_loader):            # loop over all batches
                images = images.to(device)
                labels = labels.to(device)

                y_hat = model(images)

                for i in range((y_hat).shape[0]):        # loop within a batch
                    predicted = torch.argmax(y_hat[i])
                    if predicted == labels[i].item():
                        num_correct += 1
                del images, labels, y_hat
            val_acc = 100*(num_correct/total_val_imgs)
            if val_acc > best_val_accuracy:
                best_model_found = model
            print('Val accuracy on {} validation images: {:.2f} %'.format(total_val_imgs, val_acc))

def main():
    # create data loaders
    path_to_imagenet = "/home/ian/dataset/ImageNet"                              # unsure about path?
    train_transforms, val_transforms = data_transforms('imagenet1k_basic')   # unsure about 1k basic?
    train_set, val_set = dataset(False, train_transforms, val_transforms, path_to_imagenet)
    
    batch_size = 128
    train_loader, val_loader = data_loader(False, train_set, val_set, batch_size)

    # setting CUDA, allowed to use GPU0
    gpu_id = 0
    device = f'cuda:{gpu_id}'

    # train + save model
    train_n_save(device, train_loader, val_loader)
    print("Fully trained original ResNet50 saved: original_resnet50_epoch50.pth")

if __name__ == '__main__':
    main()