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


def get_base_dir():
    if os.path.exists('/content/drive'):
        return "/content/drive/MyDrive/all_multileaf_datasets"
    return "."


class MultiLeafDataset(Dataset):
    NUM_CLASSES = 2

    def __init__(self, split, fold,
                 img_dir="resized_multileaf/images",
                 mask_dir="resized_multileaf/masks",
                 xml_dir='resized_multileaf/xmls',
                 fold_path="Folds",
                 img_size=512,
                 val_mode=True):

        if split not in ['train', 'val', 'test']:
            raise ValueError("O parâmetro 'split' deve ser 'train', 'val' ou 'test'.")

        self.split = split
        self.fold = fold
        self.img_size = img_size
        self.img_paths = []
        self.mask_paths = []
        self.xml_paths = []
        self.imgs_size = {}
        self.marker_sides = {}
        self.leaf_target_areas = {}
        self.val_fold = -1

        self.base_dir = get_base_dir()

        self.img_dir = os.path.join(self.base_dir, img_dir)
        self.mask_dir = os.path.join(self.base_dir, mask_dir)
        self.xml_dir = os.path.join(self.base_dir, xml_dir)
        self.fold_path = os.path.join(self.base_dir, fold_path)

        self.complete_spreadsheet = os.path.join(
            self.base_dir, 'final_dataset_spreadsheet.csv'
        )

        self.original_img_path = os.path.join(
            self.base_dir, 'dataset_consolidado/images'
        )

        self.comp_df = pd.read_csv(self.complete_spreadsheet)

        for filename in os.listdir(self.original_img_path):
            path = os.path.join(self.original_img_path, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            self.imgs_size[filename] = (orig_w, orig_h)

        if val_mode:
            if self.fold == 5:
                self.val_fold = 1
            else:
                self.val_fold = self.fold + 1

        if split == 'test':
            self._save_imgs(self.fold)
        elif split == 'val':
            self._save_imgs(self.val_fold)
        else:
            folds_to_complete = [f for f in range(1, 6) if f not in (self.fold, self.val_fold)]
            for f in folds_to_complete:
                self._save_imgs(f)

        unique_data = {}
        for img_path, mask_path, xml_path in zip(self.img_paths, self.mask_paths, self.xml_paths):
            filename = os.path.basename(img_path)
            unique_data[filename] = (img_path, mask_path, xml_path)

        self.img_paths = []
        self.mask_paths = []
        self.xml_paths = []

        for img_path, mask_path, xml_path in unique_data.values():
            self.img_paths.append(img_path)
            self.mask_paths.append(mask_path)
            self.xml_paths.append(xml_path)

        for path in self.xml_paths:
            tree = ET.parse(path)
            root = tree.getroot()

            pattern_side = float(root.find("pattern-side").text)
            self.marker_sides[os.path.basename(path)] = pattern_side

            invalid_area_found = False
            total_area = 0.0

            for leaf in root.findall("objects/leaf"):
                area_tag = leaf.find("dimensions/area")

                if area_tag is not None and area_tag.text is not None:
                    area_text = area_tag.text.strip()

                    try:
                        total_area += float(area_text)
                    except ValueError:
                        print(f"[WARNING] Invalid area in {os.path.basename(path)}: {area_text}")
                        invalid_area_found = True
                        break
                else:
                    invalid_area_found = True
                    break

            if invalid_area_found:
                self.leaf_target_areas[os.path.basename(path)] = -1.0
            else:
                self.leaf_target_areas[os.path.basename(path)] = total_area

        print('=========================================')
        print(f'Base dir: {self.base_dir}')
        print(f'{self.split} - {len(self.img_paths)}')
        print(f'Unique {self.split} - {len(set([os.path.basename(p) for p in self.img_paths]))}')
        print('=========================================')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        xml_path = self.xml_paths[idx]

        filename = os.path.basename(img_path)

        if filename not in self.imgs_size:
            raise KeyError(f"[ERRO] {filename} não encontrado em dataset_consolidado/images")

        orig_w, orig_h = self.imgs_size[filename]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = transforms.RandomRotation.get_params([-15, 15])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

            color_jitter = transforms.ColorJitter(brightness=0.2)
            image = color_jitter(image)

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = ((mask == 1) | (mask == 2)).long()

        pattern_side = self.marker_sides[os.path.basename(xml_path)]
        target_area = self.leaf_target_areas[os.path.basename(xml_path)]

        return image, mask, orig_w, orig_h, filename, pattern_side, target_area

    def _save_imgs(self, fold):
        current_fold = os.path.join(self.fold_path, f'fold_{fold}.csv')
        df = pd.read_csv(current_fold)

        todas_instancias = df['Group Image'].str.split(',').explode().str.strip().tolist()
        df_filtrado = self.comp_df[self.comp_df['Group'].isin(todas_instancias)]

        nomes_arquivos = df_filtrado['Current Name'].tolist()

        self.img_paths.extend([os.path.join(self.img_dir, f) for f in nomes_arquivos])

        self.mask_paths.extend([
            os.path.join(self.mask_dir, os.path.splitext(f)[0] + ".png")
            for f in nomes_arquivos
        ])

        self.xml_paths.extend([
            os.path.join(self.xml_dir, os.path.splitext(f)[0] + ".xml")
            for f in nomes_arquivos
        ])


if __name__ == '__main__':
    for f in range(1, 6):
        train_set = MultiLeafDataset('train', f, val_mode=False)
        test_set = MultiLeafDataset('test', f, val_mode=False)