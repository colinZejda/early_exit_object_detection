import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

from fasterRCNN import *
from utils import *

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # print(img.size)
        # print(coco_annotation[0]['categories'])

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = (coco_annotation[i]['bbox'][0]) * (img.size[0]/640)
            ymin = (coco_annotation[i]['bbox'][1]) * (img.size[1]/480)
            xmax = (xmin + coco_annotation[i]['bbox'][2]) * (img.size[0]/640)
            ymax = (ymin + coco_annotation[i]['bbox'][3]) * (img.size[1]/480)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append((coco_annotation[i]["category_id"]+1))

        if len(boxes) == 0:
            boxes = torch.zeros(100,4)
            labels = torch.zeros(100)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels (In my case, I only one class: target class or background)
            labels = torch.as_tensor(labels, dtype=torch.float32)
            # print(labels.shape)
            if num_objs < 100:
                boxes = torch.nn.functional.pad(boxes, (0,0,0,(100-num_objs)), value=0)
                labels = torch.nn.functional.pad(labels, (0,(100-num_objs)), value=0)
        # Tensorise img_id
        # img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        # areas = []
        # for i in range(num_objs):
        #     areas.append(coco_annotation[i]['area'])
        # areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        # my_annotation["image_id"] = img_id
        # my_annotation["area"] = areas
        # my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Resize((640,480)))
    return torchvision.transforms.Compose(custom_transforms)


# path to your own data and coco file
train_data_dir = '../../dataset/COCO/train2017'
train_coco = '../../dataset/COCO/annotations/instances_train2017.json'
train_coco_captions = '../../dataset/COCO/annotations/captions_train2017.json'

# create own Dataset
coco_dataset = COCODataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )

# coco_dataset = torchvision.datasets.CocoDetection(root=train_data_dir, annFile=train_coco)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 15

# own DataLoader
data_loader = torch.utils.data.DataLoader(coco_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4)

# print(next(iter(data_loader)))


device = torch.device('cuda:2') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

detector = TwoStageDetector((640, 480), (15, 20), 2048, 250, (2,2)).to(device)

print(detector)

# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)

def training_loop(model, learning_rate, train_dataloader, n_epochs):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for i in tqdm(range(n_epochs)):
        total_loss = 0
        i = 0
        for imgs, annotations in tqdm(train_dataloader):
            
            # print(imgs.shape)
            # print(annotations["boxes"].shape)
            # print('LABELS!:',annotations["labels"].shape)

            # forward pass
            loss = model(imgs.to(device), annotations["boxes"].to(device), annotations["labels"].to(device))
            
            if loss != -9999:
                # backpropagation
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

                optimizer.zero_grad()

                if i != 0 and i%500 == 0:
                    print(total_loss/(i+1))

                i+= 1
        
        loss_list.append(total_loss/(i+1))
        
    return loss_list

learning_rate = 1e-3
n_epochs = 5

loss_list = training_loop(detector, learning_rate, data_loader, n_epochs)

