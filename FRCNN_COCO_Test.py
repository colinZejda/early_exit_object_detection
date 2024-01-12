import os
import sys 
import torch
import torch.utils.data
import torchvision
import torchvision.ops as ops
from torchvision import models, datasets#, tv_tensors
from torchvision.transforms import v2
from PIL import Image
from pycocotools.coco import COCO

from fasterRCNN import *
from wrapper_FRCNN import *
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
            labels.append((coco_annotation[i]["category_id"]))

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
    custom_transforms.append(torchvision.transforms.PILToTensor())
    custom_transforms.append(torchvision.transforms.Resize((640,480)))
    return torchvision.transforms.Compose(custom_transforms)

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((640,480)),
        #v2.RandomPhotometricDistort(p=1),
        #v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        # v2.RandomIoUCrop(),
        #v2.RandomHorizontalFlip(p=1),
        # v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

# path to your own data and coco file
train_data_dir = '../../dataset/COCO/train2017'
train_coco = '../../dataset/COCO/annotations/instances_train2017.json'
train_coco_captions = '../../dataset/COCO/annotations/captions_train2017.json'

val_data_dir = '../../dataset/COCO/val2017'
val_coco = '../../dataset/COCO/annotations/instances_val2017.json'

# create own Dataset
# coco_dataset = COCODataset(root=train_data_dir,
#                           annotation=train_coco,
#                           transforms=get_transform()
#                           )

# val_coco_dataset = COCODataset(root=val_data_dir,
#                           annotation=val_coco,
#                           transforms=get_transform()
#                           )

coco_dataset = datasets.CocoDetection(root=train_data_dir,
                          annFile=train_coco,
                        #   transforms=get_transform()
                          transforms = transforms
                          )

val_coco_dataset = datasets.CocoDetection(root=val_data_dir,
                          annFile=val_coco,
                        #   transforms=get_transform()
                          transforms = transforms
                          )

coco_dataset = datasets.wrap_dataset_for_transforms_v2(coco_dataset, target_keys=("boxes", "labels"))
val_coco_dataset = datasets.wrap_dataset_for_transforms_v2(val_coco_dataset, target_keys=("boxes", "labels"))

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 10
accum = 4

# own DataLoader
data_loader = torch.utils.data.DataLoader(coco_dataset,
                                          batch_size=train_batch_size,
                                          collate_fn = collate_fn,
                                          shuffle=True,
                                          num_workers=4)

val_loader = torch.utils.data.DataLoader(val_coco_dataset,
                                          batch_size=train_batch_size,
                                          collate_fn = collate_fn,
                                          shuffle=True,
                                          num_workers=4)

# (image, anno) = next(iter(data_loader))
# # print(len(image), image[0].shape)
# print(anno)


device = torch.device('cuda:1') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#detector = TwoStageDetector((640, 480), (15, 20), 2048, 100, (2,2)).to(device)
# detector.load_state_dict(torch.load('FRCNN_model.pth',map_location=device))
detector = FRCNN_wrapper(transforms).to(device)

for name, param in detector.named_parameters():
    if param.requires_grad:
        print(name)#, param.data)
# print(detector)

# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)

def training_loop(model, learning_rate, train_dataloader, n_epochs, val_loader, batch_size):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_list = []
    
    for epoch in range(n_epochs):#tqdm(range(n_epochs)):
        batch_loss = 0
        total_loss = 0
        i = 0
        accum_counter = 1
        printed = False
        print('=== Training epoch: ', epoch, '===')
        for data in train_dataloader: #tqdm(train_dataloader):
            
            # print(imgs.shape)
            # print(annotations["boxes"].shape)
            # print('LABELS!:',annotations["labels"].shape)

            images = list(image.to(device) for image in data[0])
            targets = [{k: v.to(device) for k, v in t.items()} for t in data[1]]

            # forward pass
            loss = model(images, targets)
            batch_loss += loss['loss_classifier'] + loss['loss_box_reg']

            if loss != -9999:
                # backpropagation
                batch_loss.backward()
                total_loss += batch_loss.item()/accum
                batch_loss = 0

                
                if accum_counter == accum:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 1
                    i+=1
                    printed = False
                else:
                    accum_counter += 1

                if i != 0 and i%125 == 0 and printed == False:
                    print('loss:', total_loss/(i+1))
                    sys.stdout.flush()
                    printed = True

                #i+= 1
        
        loss_list.append(total_loss/(i+1))

        torch.save(model.state_dict(), 'FRCNN_model-2.pth')
        print('Total Loss:', total_loss/(i+1))
        # model.eval()

        # TP = 0
        # FP = 0
        # FN = 0
        # IOU = 0
        # print('=== Validation epoch: ', epoch, '===')
        # for imgs, annotations in val_loader:#tqdm(val_loader):
        #     proposals_final, conf_scores_final, classes_final = model.inference(imgs.to(device))
        #     annotations['labels'] = annotations['labels'].to(device)
        #     annotations['boxes'] = annotations['boxes'].to(device)

            

        #     no_IOU = False
        #     # IOU_holder = 0
        #     for index in range(len(classes_final)):
        #         # print('class:',classes_final[index])
        #         # print('gt_class', annotations["labels"][index])
        #         # print('conf',conf_scores_final[index])

        #         # print(classes_final[index].shape)
        #         # print(proposals_final[index].shape)
        #         # print(annotations["labels"][index].shape)
        #         # print(annotations["boxes"][index].shape)
        #         model_instances = classes_final[index].shape[0]
        #         gt_instances = annotations["labels"][index].shape[0]
        #         max_instances = max(model_instances, gt_instances)

        #         # IOU_holder = ops.box_iou(proposals_final[index], annotations['boxes'][index])
        #         # print(IOU_holder)

        #         for instance in range(max_instances):
        #             if instance >= model_instances:
        #                 FN += 1 if annotations["labels"][index][instance] != 0 else 0 
        #                 no_IOU = True
        #             elif instance >= gt_instances:
        #                 FP += 1 if classes_final[index][instance] != 0 else 0
        #                 no_IOU = True
        #             elif classes_final[index][instance] == annotations["labels"][index][instance] and annotations["labels"][index][instance] != -1 and classes_final[index][instance] != -1:
        #                 TP += 1
        #             elif classes_final[index][instance] != annotations["labels"][index][instance] and classes_final[index][instance] == -1:
        #                 FN += 1
        #                 no_IOU = True
        #             elif classes_final[index][instance] != annotations["labels"][index][instance]:
        #                 FP += 1

        #             # if no_IOU:
        #             #     no_IOU = False
        #             #     IOU_holder = 0
        #             # else:
        #             #     IOU_holder = ops.box_iou(proposals_final[index][instance], annotations['boxes'][index][instance]).item()
                    
        #             # IOU += IOU_holder

        # prec = (TP)/(TP+FP)
        # recall = (TP)/(TP+FN)
        # print(prec)
        # print(recall)
        # print(TP, '--', FP, '--', FN)
        # sys.stdout.flush()

        
    return loss_list

learning_rate = 1e-3
n_epochs = 100

loss_list = training_loop(detector, learning_rate, data_loader, n_epochs, val_loader, train_batch_size)

