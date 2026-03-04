import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import glob
import pandas as pd
import re

class MultiLeafDataset(Dataset):
    NUM_CLASSES = 3
    def __init__(self, args, split, fold):
        self.args = args
        self.split = split
        self.img_dir = 'resized_dataset/dataset_processado/resized_fixed_512'
        self.seg_dir = 'resized_dataset/mascaras_processadas/resized_fixed_512'
        self.image_paths = []
        self.segmentation_paths = []
 
        # Carrega CSV de folds
        fold_csv_path = os.path.join(args.folds_path, 'Folds_Divisao_v20260303_162811_max4.csv.csv')
        df = pd.read_csv(fold_csv_path)
        self.group_to_images = {}

        # Monta dicionário group -> instâncias
        for _, row in df.iterrows():
            group_name = row["Group Name"]
            instances = row["Instances"]
            images = [x.strip() for x in instances.split(",")]
            self.group_to_images[group_name] = images

        # Define grupos de treino ou teste
        if 'Fold' in df.columns:
            if split == 'train':
                groups = df[df["Fold"] != fold]["Group Name"].tolist()
            else:
                groups = df[df["Fold"] == fold]["Group Name"].tolist()
        else:
            groups = df["Group Name"].tolist()

        for group in groups:
            for instance_id in self.group_to_images[group]:
                pattern = re.compile(rf"{re.escape(instance_id)}(_|\.|$)", re.IGNORECASE)
                all_images = glob.glob(os.path.join(self.img_dir, "*.jpg"))

                found_images = [img for img in all_images if pattern.search(os.path.basename(img))]
                
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

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seg_path = self.segmentation_paths[idx]

        image = Image.open(img_path).convert('RGB')
        image = TF.resize(image, (512, 512))
        image = TF.to_tensor(image)

        segmentation = Image.open(seg_path).convert('L')
        segmentation = TF.resize(segmentation, (512, 512), interpolation=Image.NEAREST)
        segmentation = torch.as_tensor(np.array(segmentation), dtype=torch.long)

        return image, segmentation