import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
from modeling.deeplab_seg import DeepLab

# ==========================
# Configurações e Caminho Único
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "crossval_models/best_model_fold_5.pth"
# Selecionando a primeira imagem da sua lista para destaque
SINGLE_IMAGE_PATH = "Orange_5_pat_6cm_dist_20cm_OneVision.jpg"

# Preprocessamento (Padrão ImageNet conforme seu script)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

COLORS = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]]) # BG, Marker, Leaf

# ==========================
# Carregar Modelo e Inferência
# ==========================
model = DeepLab(num_classes=3, backbone="xception", output_stride=16)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict)
model.to(DEVICE).eval()

# Processar imagem
img_raw = Image.open(SINGLE_IMAGE_PATH).convert("RGB")
input_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Preparar visualização
img_np = np.array(img_raw.resize((512, 512)))
mask_colored = COLORS[pred].astype(np.uint8)
# Overlay com 50% de transparência para análise técnica
overlay = (0.5 * img_np + 0.5 * mask_colored).astype(np.uint8)

# ==========================
# Visualização de Destaque
# ==========================
plt.figure(figsize=(18, 6))

# Subplot 1: Original
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("1. Imagem Original", fontsize=14, pad=10)
plt.axis("off")

# Subplot 2: Máscara Pura (Segmentação Genética)
plt.subplot(1, 3, 2)
plt.imshow(mask_colored)
plt.title("2. Segmentação (Predição)", fontsize=14, pad=10)
plt.axis("off")

# Subplot 3: Overlay (Análise de Precisão)
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("3. Sobreposição (Overlay)", fontsize=14, pad=10)
plt.axis("off")

plt.tight_layout()
plt.show()