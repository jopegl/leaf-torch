import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = '/home/jope/Code/ic/leaf-torch/resized_dataset/dataset_processado/resized_fixed_512/4_Corn_pat_2cm_center_dist_50cm_MotoC.jpg'
mask_path = "/home/jope/Code/ic/leaf-torch/resized_dataset/mascaras_processadas/resized_fixed_512/4_Corn_pat_2cm_center_dist_50cm_MotoC.png"

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"Imagem não encontrada: {img_path}")

if mask is None:
    raise ValueError(f"Máscara não encontrada: {mask_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Garantir binária
mask_bin = (mask > 0).astype(np.uint8)

# Criar camada vermelha
overlay = img.copy()
overlay[mask_bin == 1] = [255, 0, 0]  # vermelho puro

# Misturar
result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

plt.figure(figsize=(6,6))
plt.imshow(result)
plt.title("Overlay")
plt.axis("off")
plt.show()