import os
import glob
import re
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MultiLeafDataset(Dataset):
    NUM_CLASSES = 3

    def __init__(self, split, fold,
                 img_dir="resized_dataset/dataset_processado/resized_fixed_512",
                 mask_dir="resized_dataset/mascaras_processadas/resized_fixed_512",
                 fold_path="Folds",
                 img_size=512):
        
        if split not in ['train', 'val', 'test']:
            raise ValueError("O parâmetro 'split' deve ser 'train', 'val' ou 'test'.")

        self.split = split
        self.fold = fold
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.samples = []

        # --- LÓGICA DE FOLDS ---
        todos_csvs = glob.glob(os.path.join(fold_path, "*.csv"))
        csvs_folds = [f for f in todos_csvs if "leafmap" not in os.path.basename(f).lower()]
        
        # Garante a ordem correta dos Folds (1, 2, 3...)
        csvs_folds.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) if re.search(r'\d+', os.path.basename(x)) else 0)
        
        num_folds = len(csvs_folds)
        if num_folds == 0:
            raise ValueError(f"Nenhum CSV de Fold encontrado na pasta: {fold_path}")

        # Índices dos Folds (0-indexed)
        idx_test = (fold - 1) % num_folds
        idx_val = fold % num_folds 

        grupos_por_fold = {}
        todos_grupos_mapeados = set()

        # LEITURA LIMPA E DIRETA DO CSV
        for i, csv_file in enumerate(csvs_folds):
            try:
                df = pd.read_csv(csv_file, header=4, encoding='utf-8')
                
                # Limpa os nomes das colunas removendo espaços extras
                df.columns = df.columns.str.strip()
                
                # Agora procura pela coluna correta: 'Group Image'
                if 'Group Image' in df.columns:
                    grupos = df['Group Image'].dropna().astype(str).str.strip().tolist()
                    grupos_por_fold[i] = set(grupos)
                    todos_grupos_mapeados.update(grupos)
                else:
                    print(f"[ERRO] Coluna 'Group Image' não encontrada no arquivo {os.path.basename(csv_file)}. Colunas achadas: {df.columns.tolist()}")
                    grupos_por_fold[i] = set()

            except Exception as e:
                print(f"Erro ao ler o CSV {os.path.basename(csv_file)}: {e}")
                grupos_por_fold[i] = set()

        grupos_test = grupos_por_fold[idx_test]
        grupos_val = grupos_por_fold[idx_val]

        # --- SEPARAÇÃO DAS IMAGENS ---
        arquivos_imagens = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Ordena do nome maior para o menor para evitar que 'corn_1' dê match em 'corn_10'
        grupos_ordenados = sorted(list(todos_grupos_mapeados), key=len, reverse=True)

        for nome_img in arquivos_imagens:
            grupo_da_imagem = None
            for g in grupos_ordenados:
                if nome_img.startswith(g + "_"):
                    grupo_da_imagem = g
                    break

            if grupo_da_imagem in grupos_test:
                destino = "test"
            elif grupo_da_imagem in grupos_val:
                destino = "val"
            else:
                destino = "train"

            if destino == self.split:
                caminho_img = os.path.join(img_dir, nome_img)
                nome_base = os.path.splitext(nome_img)[0]
                caminho_mask = os.path.join(mask_dir, f"{nome_base}.png")
                
                if os.path.exists(caminho_mask):
                    self.samples.append((caminho_img, caminho_mask))

        print(f"Dataset carregado -> Split: {self.split.upper():<5} | Fold: {self.fold} | Amostras: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = TF.resize(image, (self.img_size, self.img_size))
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        mask = Image.open(mask_path).convert("L")
        mask = TF.resize(mask, (self.img_size, self.img_size),
                         interpolation=Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask