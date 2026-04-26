import os
import glob
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from collections import defaultdict
import csv


def get_base_dir():
    if os.path.exists('/content/drive'):
        return "/content/drive/MyDrive/all_multileaf_datasets"
    return "."


class MultiLeafDataset(Dataset):
    def __init__(self, split, fold,
                 dataset_root="multileaf_dataset",
                 fold_csv="folds.csv",
                 img_size=512,
                 val_mode=True,
                 num_classes=2):

        if split not in ['train', 'val', 'test']:
            raise ValueError("split deve ser train, val ou test")

        self.split = split
        self.fold = fold
        self.img_size = img_size
        self.NUM_CLASSES = num_classes

        self.img_paths = []

        # monta folds
        folds = self._build_fold_image_dict(fold_csv, dataset_root)

        if self.split == 'train':
            self.img_paths = folds[self.fold]

        elif self.split == 'val':
            self.val_fold = 5 if self.fold != 5 else 1
            self.img_paths = folds[self.val_fold]

        else:
            self.img_paths = []
            for f in range(1, 6):
                if f != self.fold and f != getattr(self, "val_fold", -1):
                    self.img_paths += folds[f]


    def __len__(self):
        return len(self.img_paths)


    def _get_mask_path(self, img_path):
        # multileaf_dataset/Bean/images/x.jpg
        # -> multileaf_dataset/Bean/masks/x.png
        return (
            img_path
            .replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)
            .rsplit(".", 1)[0] + ".png"
        )


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self._get_mask_path(img_path)

        filename = os.path.basename(img_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # =========================
        # AUGMENTATION (train)
        # =========================
        if self.split == "train":
            scale = np.random.uniform(0.75, 1.25)
            new_w, new_h = int(self.img_size * scale), int(self.img_size * scale)

            image = TF.resize(image, (new_h, new_w))
            mask = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

            if new_h > self.img_size and new_w > self.img_size:
                top = np.random.randint(0, new_h - self.img_size)
                left = np.random.randint(0, new_w - self.img_size)
            else:
                top, left = 0, 0

            image = TF.crop(image, top, left, self.img_size, self.img_size)
            mask = TF.crop(mask, top, left, self.img_size, self.img_size)

            if np.random.rand() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            jitter = transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
                saturation=0.08
            )
            image = jitter(image)

            angle = np.random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        else:
            image = TF.resize(image, (self.img_size, self.img_size))
            mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        # =========================
        # TO TENSOR
        # =========================
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        if self.NUM_CLASSES == 2:
            mask = ((mask == 1) | (mask == 2)).long()
        elif self.NUM_CLASSES == 3:
            mask = mask.long()
        else:
            raise ValueError("NUM_CLASSES deve ser 2 ou 3")

        return image, mask, filename


    def _build_fold_image_dict(self, csv_path, dataset_root):
        fold_dict = defaultdict(list)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                fold = int(row["Fold"])
                image_name = row["Image Name"]
                specie = row["Specie"].strip()

                img_path = os.path.join(
                    dataset_root,
                    specie,
                    "images",
                    image_name
                )

                fold_dict[fold].append(img_path)

        return dict(fold_dict)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("\n=== DEBUG DATASET ===")

    for f in range(1, 6):
        print(f"\n--- FOLD {f} ---")

        dataset = MultiLeafDataset('train', f)

        print(f"Total imagens no fold {f}: {len(dataset)}")

        for i in range(min(5, len(dataset))):
            img, mask, filename = dataset[i]

            print(f"\n[{i}] {filename}")

            # =====================
            # checks básicos
            # =====================
            print("Image shape:", img.shape)
            print("Mask shape:", mask.shape)

            unique_vals = torch.unique(mask)
            print("Mask values:", unique_vals.tolist())

            if mask.sum() == 0:
                print("⚠️ ALERTA: máscara vazia!")

            # =====================
            # sanity check shapes
            # =====================
            assert img.shape[1:] == (512, 512), "Imagem não está 512x512"
            assert mask.shape == (512, 512), "Máscara não está 512x512"

            # =====================
            # visual debug (opcional)
            # =====================
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.title("Image")
            plt.imshow(img_np)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Mask")
            plt.imshow(mask.numpy(), cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        print("\n✔ Fold OK\n")