import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import csv

leaf_dataset_path = os.path.join('real_dataset', 'LSID-Beans')

class LeafSegmentation(Dataset):
    NUM_CLASSES = 3 

    def __init__(self, args, split="train", cross_val_folder = None):
        self.args = args
        self.split = split

        self.image_paths = []
        self.mask_paths = []
        self.area_paths = []

        self.crop_size = getattr(args, 'crop_size', 512)

        base_dir = os.path.join(leaf_dataset_path)
        
        if cross_val_folder is not None:
            print("\033[93m[INFO] Cross-validation mode enabled\033[0m")
            leaf_ids_train, leaf_ids_test = self._load_leaf_ids_from_csv(cross_val_folder)
            leaf_ids = leaf_ids_train if split == 'train' else leaf_ids_test

            for num in leaf_ids:

                num = str(int(num)).zfill(3)

                images_dir = os.path.join(base_dir, num, 'images')
                area_dir = os.path.join(base_dir, num, 'area')
                segmentation_dir = os.path.join(base_dir, num, 'segmentation')

                if not (
                    os.path.isdir(images_dir) and
                    os.path.isdir(area_dir) and
                    os.path.isdir(segmentation_dir)
                ):
                    print(f"[WARN] Folder not found for leaf_id {num}")
                    continue

                images = sorted(os.listdir(images_dir))
                areas = sorted(os.listdir(area_dir))
                masks = sorted(os.listdir(segmentation_dir))

                for i, a, s in zip(images, areas, masks):
                    self.image_paths.append(os.path.join(images_dir, i))
                    self.area_paths.append(os.path.join(area_dir, a))
                    self.mask_paths.append(os.path.join(segmentation_dir, s))

        else:
            dir_with_splits = os.path.join(base_dir, split)
            leaf_ids = sorted(os.listdir(dir_with_splits))

            for leaf_id in leaf_ids:
                leaf_path = os.path.join(dir_with_splits, leaf_id)
                self.image_dir = os.path.join(leaf_path,"images")
                self.mask_dir = os.path.join(leaf_path,"segmentation")
                self.area_dir = os.path.join(leaf_path,'area')

                images = sorted(os.listdir(self.image_dir))
                masks = sorted(os.listdir(self.mask_dir))
                areas = sorted(os.listdir(self.area_dir))

                for image, mask, area in zip(images, masks, areas):
                    self.image_paths.append(os.path.join(self.image_dir, image))
                    self.mask_paths.append(os.path.join(self.mask_dir, mask))
                    self.area_paths.append(os.path.join(self.area_dir, area))

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        area_path = self.area_paths[index]

        image = Image.open(img_path).convert('RGB')
        mask_data = np.fromfile(mask_path, dtype=np.int8)
        mask_data = mask_data.reshape((self.crop_size, self.crop_size))
        mask = torch.from_numpy(mask_data)

        image = self.image_transform(image)
        area_data = np.fromfile(area_path, dtype=np.float32) 
        area_data = area_data.reshape((self.crop_size, self.crop_size))  
        area = torch.from_numpy(area_data)
        area = area.unsqueeze(0)

        return image, mask, area
    
    def _load_leaf_ids_from_csv(self, split_folder):
        train_csv = os.path.join(split_folder, 'train.csv')
        test_csv = os.path.join(split_folder, 'test.csv')
        leaf_ids_train = []
        leaf_ids_test = []
        with open(train_csv, newline = '') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].lower() == 'id':
                    continue

                leaf_id = row[0].split('_')[1]
                if leaf_id.endswith('_area'):
                    leaf_id = leaf_id.replace('_area', '')
                leaf_ids_train.append(leaf_id)
        
        with open(test_csv, newline = '') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].lower() == 'id':
                    continue

                leaf_id = row[0].split('_')[1]
                if leaf_id.endswith('_area'):
                    leaf_id = leaf_id.replace('_area', '')
                leaf_ids_test.append(leaf_id)
        
        leaf_ids_train = sorted(set(leaf_ids_train))
        leaf_ids_test = sorted(set(leaf_ids_test))

        return leaf_ids_train, leaf_ids_test
