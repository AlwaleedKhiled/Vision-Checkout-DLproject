#imports
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import json


#
class CustomAugmentationClass(Dataset):
    '''
    This class is a custom implementation of a PyTorch Dataset for object detection tasks.
    It reads images and their corresponding annotations from a specified directory and JSON file,
    applies transformations using Albumentations, and prepares the data in a format suitable for training object detection models.
    '''

    #Init function.
    def __init__(self,images_dir, json_path, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        # Load the JSON file:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = data['categories']


        #pre-map the annotation to image_ID for faster access later O(n) instead of looping later:
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']

            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            

    #len function:
    def __len__(self):
        return len(self.images)

    #getitem function:
    def __getitem__(self,idx):
        img_info = self.images[idx] # Image entries: {'file_name': '038900004095_camera0-13.jpg', 'width': 2592, 'height': 1944, 'id': 0}
        img_id = img_info['id']

        img_name = img_info['file_name']

        full_path = os.path.join(self.images_dir, img_name)

        #1- Albumentations only accepts numpy arrays
        image = Image.open(full_path).convert('RGB')
        image = np.array(image)

        #2- time to extract the features (annotations):

        #retrieve the pre-stored anns
        anns = self.img_to_anns.get(img_id, [])

        boxes,labels = [],[]

        for ann in anns:
            #convert COCO to Pytorch: [x,y,w,h] -> [x1,y1,x2,y2]

            x,y,w,h = ann['bbox']
            boxes.append([x, y, (x+w), (y+h)])
            labels.append(ann['category_id'])

        #3- apply transformations with Albumentations.
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, category_ids=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']



        #4- convert EVERYTHING to Tensors (after Transformation)
        #and handle empty boxes just in case:
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        
        #5- Define target (as a Dictionary !!!)
        target = {
            'boxes':boxes,
            'labels':labels,
            'image_id':torch.tensor([img_id])
        }


        return image, target
 

def collate_fn(batch):
    """
    since each image has a various number of objects (boudning boxes),
    This function helps ensure that they are grouped as a list rather than a single tensor,
    therefore allowing more than 1 image in a batch
    """
    return tuple(zip(*batch))