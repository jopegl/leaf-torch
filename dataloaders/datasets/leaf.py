import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()
leaf_dataset_path = os.getenv('LEAF_DATASET_PATH', 'complete_leaf_dataset')

class LeafSegmentation(Dataset):
    NUM_CLASSES = 3 

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split

        self.image_paths = []
        self.mask_paths = []
        self.area_paths = []

        base_dir = os.path.join(leaf_dataset_path)
        
        dir_with_splits = os.path.join(base_dir, split)
        for leaf_id in sorted(os.listdir(dir_with_splits)):
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
        mask_data = mask_data.reshape((512, 512))
        mask = torch.from_numpy(mask_data)

        image = self.image_transform(image)
        area_data = np.fromfile(area_path, dtype=np.float32) # mudar para float32
        area_data = area_data.reshape((512, 512))  
        area = torch.from_numpy(area_data)
        area = area.unsqueeze(0)

        return image, mask, area
