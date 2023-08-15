import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

class KAIST_Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split_T, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored / json 파일 있는 경로를 전달해줌 -> KAIST_PD_OUTPUT
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split_T = split_T.upper() # train / test 여부 결정
             # lwir / visible 여부 결정

        assert self.split_T in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # '/home/urp0/workspace/KAIST_PD_OUTPUT'  - data folder
        
        if self.split_T=='TRAIN':
            # Read data files
            #import pdb;pdb.set_trace()
            with open(os.path.join(data_folder,'visible_'+ self.split_T + '_images.json'), 'r') as j:
                self.rgb_images = json.load(j)
            with open(os.path.join(data_folder,'lwir_'+ self.split_T + '_images.json'), 'r') as j:
                self.thermal_images = json.load(j)
            #이미지를 로드
            with open(os.path.join(data_folder,'KAIST'+'_'+ self.split_T + '_objects.json'), 'r') as j:
                self.objects = json.load(j)
                #이미지에 맞는 objects 를 로드

        if self.split_T=='TEST':
            
            with open(os.path.join(data_folder,'visible_'+ self.split_T + '_images.json'), 'r') as j:
                self.rgb_images = json.load(j)
            with open(os.path.join(data_folder,'lwir_'+ self.split_T + '_images.json'), 'r') as j:
                self.thermal_images = json.load(j)
                #이미지를 로드
            with open(os.path.join(data_folder,'KAIST'+'_'+ self.split_T + '_objects.json'), 'r') as j:
                self.objects = json.load(j)
                #이미지에 맞는 objects 를 로드
        
        
        

        print(len(self.rgb_images),len(self.objects))


        assert len(self.rgb_images) == len(self.objects) == len(self.thermal_images)

    def __getitem__(self, i):
        # Read image
        rgb_image = Image.open(self.rgb_images[i], mode='r')
        rgb_image = rgb_image.convert('RGB')
        thermal_image = Image.open(self.thermal_images[i], mode='r')
        thermal_image = thermal_image.convert('RGB')

       
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['category_id'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['is_crowd'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]
        #import pdb;pdb.set_trace()
        
        rgb_image,thermal_image, boxes, labels, difficulties = transform(rgb_image,thermal_image, boxes, labels, difficulties, split=self.split_T)
        
        img_list=[]
        img_list.append(rgb_image)
        img_list.append(thermal_image)
        return img_list, boxes, labels, difficulties

    def __len__(self):
        return len(self.rgb_images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        rgb_images = list()
        thermal_images=list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            rgb_images.append(b[0][0])
            thermal_images.append(b[0][1])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        rgb_images = torch.stack(rgb_images, dim=0)
        thermal_images=torch.stack(thermal_images,dim=0)
        
        
        
        img_list=[]
        img_list.append(rgb_images)
        img_list.append(thermal_images)

        return img_list, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

