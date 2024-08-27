import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
from collections import defaultdict
import numpy as np
import cv2
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random

# ------ Single-Label Image Datasets ------
class SingleLabelImageFolder(Dataset):
    def __init__(self, root, cls_num, transform=None, target_transform=None, modality='fundus', if_semi=False, test_file=None):
        super(SingleLabelImageFolder, self).__init__()
        self.cls_num = cls_num
        self.modality = modality
        if test_file is None:
            file = 'large9cls.txt'
            path_file = os.path.join(root, file)
        else:
            path_file = test_file
        self.root = '/'.join(os.path.split(root)[:-1])
        self.transform = transform
        self.target_transform = target_transform
        
        self.img2labels = {}
        self.imgs = []
        
        with open(path_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if test_file is not None:
                    img_name = line.strip().split('\t')[0]
                    label = int(line.strip().split('\t')[1])
                else:
                    img_name = line.strip().split(' ')[0]
                    label = int(line.strip().split(' ')[1])
          
                self.imgs.append(img_name)
                self.img2labels[img_name] = label

    def __getitem__(self, index):
        if self.modality == 'fundus':
            img_path = os.path.join(self.root, 'train', 'ImageData', 'cfp-clahe-224x224', self.imgs[index] + '.png')
        if self.modality == 'oct':
            img_path = os.path.join(self.root, 'train', 'ImageData', 'oct-filter-448x448', self.imgs[index] + '.png')
        img = Image.open(img_path).convert('RGB')
        img_name = os.path.split(img_path)[-1][:-4]
        # Get the labels for the corresponding image
        label = self.img2labels[self.imgs[index]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, img_name
    
    def __len__(self):
        return len(self.imgs)
    
    def label_statistics(self):
        cls_count = np.zeros(self.cls_num).astype(np.int64)

        for i, label in self.img2labels.items():
            cls_count[label] += 1
        return cls_count
            
    def label_weights_for_balance(self):
        cls_count = self.label_statistics()
        labels_weight_list = []
        for i, label in self.img2labels.items():
            weight = 1 / cls_count[label]
            labels_weight_list.append(weight)
        return labels_weight_list
    

class MultiModalSingleImageFolder(Dataset):
    def __init__(self, fundus_root, oct_root, cls_num, mode='train', transform=None, target_transform=None, transform_oct=None, if_semi=False):
        super(MultiModalSingleImageFolder, self).__init__()
        fundus_file = 'large9cls.txt'
        oct_file = 'large9cls.txt'

        fundus_path_file = os.path.join(fundus_root, fundus_file)
        oct_path_file = os.path.join(oct_root, oct_file)
        
        self.cls_num = cls_num
        self.mode = mode
        self.fundus_root = '/'.join(os.path.split(fundus_root)[:-1])
        self.oct_root = '/'.join(os.path.split(oct_root)[:-1])
        self.transform = transform
        self.target_transform = target_transform
        self.transform_oct = transform_oct
        
        self.f_img2labels = {}
        self.f_imgs = []
        self.o_labels2img = defaultdict(list)
        self.o_imgs = []
        
        with open(fundus_path_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fundus_img_name = line.strip().split(' ')[0]
                fundus_labels = int(line.strip().split(' ')[1])
                
                self.f_imgs.append(fundus_img_name)
                self.f_img2labels[fundus_img_name] = fundus_labels
        
        with open(oct_path_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                oct_img_name = line.strip().split(' ')[0]
                oct_labels = int(line.strip().split(' ')[1])
                
                self.o_imgs.append((oct_img_name, oct_labels))
                self.o_labels2img[oct_labels].append(oct_img_name)

    def __getitem__(self, index):
        f_img_path = os.path.join(self.fundus_root, 'train', 'ImageData', 'cfp-clahe-224x224', self.f_imgs[index] + '.png')
        f_img = Image.open(f_img_path).convert('RGB')
        f_img_name = os.path.split(f_img_path)[-1][:-4]
        f_labels = self.f_img2labels[self.f_imgs[index]]
        if self.transform is not None:
            f_img = self.transform(f_img)
        if self.target_transform is not None:
            f_labels = self.target_transform(f_labels)

        if self.mode != 'test':
            if f_labels in self.o_labels2img:
                o_img_name = random.sample(self.o_labels2img[f_labels], k=1)[0]
                o_labels = f_labels
            else:
                rand_k = random.randint(0, len(self.o_imgs)-1)
                o_img_name = self.o_imgs[rand_k][0]
                o_labels = self.o_imgs[rand_k][1]
            
            o_img_path = os.path.join(self.oct_root, 'train', 'ImageData', 'oct-filter-448x448', o_img_name + '.png')
            o_img = Image.open(o_img_path).convert('RGB')
            
            if self.transform_oct is not None:
                o_img = self.transform_oct(o_img)
            if self.target_transform is not None:
                o_labels = self.target_transform(o_labels)
            
            return (f_img, o_img), (f_labels, o_labels), (f_img_name, o_img_name)
        
        else:
            return f_img, f_labels, f_img_name

    def __len__(self):
        return len(self.f_imgs)
    
    def label_statistics(self):
        cls_count = np.zeros(self.cls_num).astype(np.int64)

        for i, label in self.f_img2labels.items():
            cls_count[label] += 1
        return cls_count
            
    def label_weights_for_balance(self):
        cls_count = self.label_statistics()
        labels_weight_list = []
        for i, label in self.f_img2labels.items():
            weight = 1 / cls_count[label]
            labels_weight_list.append(weight)
        return labels_weight_list


def build_dataset_single(mode, args, transform=None, mod='fundus', test_file=None):
    if transform is None:
        transform = build_transform(mode, args)
    root = os.path.join(args.data_path, mode)
    dataset = SingleLabelImageFolder(root, args.n_classes, transform=transform, modality=mod, test_file=test_file)
    return dataset



def build_dataset_multimodal_single(mode, args, transform=None):
    if transform is None:
        transform = build_transform(mode, args)
        # transform_oct = build_transform('test', args)
        transform_oct = build_transform(mode, args)
    f_root = os.path.join(args.data_path, mode)
    o_root = os.path.join(args.data_path_oct, mode)
    dataset = MultiModalSingleImageFolder(f_root, o_root, args.n_classes, mode, transform=transform, transform_oct=transform_oct)
    return dataset


def build_transform(mode, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if mode == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size == 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
