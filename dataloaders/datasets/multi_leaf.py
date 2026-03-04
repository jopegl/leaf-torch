import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn.functional as F
import csv
import pandas as pd
import cv2
import glob

class MultiLeafDataset(Dataset):
    def __init__(self, args, split, fold, img_dir, seg_dir):
        self.args = args
        self.split = split
        self.image_paths = []
        self.segmentation_paths = []
        self.img_dir = img_dir
        self.seg_dir = seg_dir

        fold_csv_path = f'{args.folds_path}/Folds_Divisao_v20260303_162811_max4.csv.csv'
        df = pd.read_csv(fold_csv_path)
        self.group_to_images = {}

        for _,row in df.iterrows():
            group_name = row["Group Image"]
            instances = row["Instances"]
            images = [x.strip() for x in instances.split(",")]
            self.group_to_images[group_name] = images

        if 'Fold' in df.columns:
            if split == 'train':
                groups = df[df["Fold"] != fold]["Group Image"].tolist()
            else:
                groups = df[df["Fold"] == fold]["Group Image"].tolist()
        else:
            groups = df["Group Image"].tolist()

        for group in groups:
            for instance_id in self.group_to_images[group]:
                pattern = os.path.join(self.img_dir, f"*{instance_id}*.jpg")
                found_images = glob.glob(pattern)

                if not found_images:
                    print(f"[WARN] Nenhuma imagem encontrada para: {instance_id}")
                    continue

                for img_path in found_images:
                    mask_name = os.path.basename(img_path).replace(".jpg", ".png")
                    mask_path = os.path.join(self.seg_dir, mask_name)

                    if not os.path.exists(mask_path):
                        print(f"[WARN] Máscara não encontrada para: {mask_name}")
                        continue

                    self.image_paths.append(img_path)
                    self.segmentation_paths.append(mask_path)

        print(f"[INFO] {split} - {len(self.image_paths)} imagens carregadas para fold {fold}")


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, key):
        img_path = self.image_paths[key]
        seg_path = self.segmentation_paths[key]

        image = Image.open(img_path).convert('RGB')
        image = TF.resize(image, (512, 512)) 
        image = TF.to_tensor(image)   

        segmentation = Image.open(seg_path).convert('L')
        segmentation = TF.resize(segmentation, (512, 512), interpolation=Image.NEAREST) 
        segmentation = torch.as_tensor(np.array(segmentation), dtype=torch.long)  

        return image, segmentation


