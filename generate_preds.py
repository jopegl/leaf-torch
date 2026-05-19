import torch
from collections import defaultdict
import csv
import os
from modeling.deeplab_seg import DeepLab
import cv2
import torchvision.transforms.functional as TF
import numpy as np
import time

MODEL_ALL_PATH = {
    1: 'pos-correcoes/crossval_models/best_model_fold_1.pth',
    2: 'pos-correcoes/crossval_models/best_model_fold_2.pth',
    3: 'pos-correcoes/crossval_models/best_model_fold_3.pth',
    4: 'pos-correcoes/crossval_models/best_model_fold_4.pth',
    5: 'pos-correcoes/crossval_models/best_model_fold_5.pth'
}

# =========================
# BUILD DICT
# =========================
def build_fold_image_dict(csv_path, dataset_root):
    fold_dict = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            fold = int(row["Fold"])
            image_name = row["Image Name"]
            specie = row["Specie"].strip()

            img_path = os.path.join(dataset_root, specie, "images", image_name)
            fold_dict[fold].append(img_path)

    return dict(fold_dict)

# =========================
# PREPROCESS
# =========================
def preprocess_image(img):
    img_resized = cv2.resize(img, (512, 512))
    img_resized = img_resized.astype(np.float32) / 255.0

    img_tensor = torch.tensor(img_resized).permute(2, 0, 1)

    img_tensor = TF.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return img_tensor.unsqueeze(0)

@torch.no_grad()
def predict(model, img_tensor):
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1)
    return pred.squeeze(0).cpu().numpy()

# =========================
# OVERLAY
# =========================
def create_overlay(image, mask):
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 1] = mask * 255  # verde

    overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
    return overlay

# =========================
# MAIN
# =========================
print("\n🚀 Iniciando inferência...\n")

fold_img_dict = build_fold_image_dict('folds.csv', 'pos-correcoes/multileaf_dataset')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# cria pastas raiz
os.makedirs("preds", exist_ok=True)
os.makedirs("sobreposicoes", exist_ok=True)

for fold in range(1, 6):

    print(f"\n==============================")
    print(f"📦 Fold {fold}")
    print(f"==============================")

    # cria pastas do fold
    pred_fold_dir = os.path.join("preds", f"fold_{fold}")
    overlay_fold_dir = os.path.join("sobreposicoes", f"fold_{fold}")

    os.makedirs(pred_fold_dir, exist_ok=True)
    os.makedirs(overlay_fold_dir, exist_ok=True)

    model = DeepLab(
        num_classes=2,
        backbone='xception',
        output_stride=16,
        sync_bn=None,
        freeze_bn=True
    ).to(device)

    state_dict = torch.load(MODEL_ALL_PATH[fold], map_location=device)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    imgs = fold_img_dict[fold]
    print(f"🖼️ Total de imagens: {len(imgs)}")

    start_fold = time.time()

    for i, img_path in enumerate(imgs):

        if i % 20 == 0:
            print(f"➡️ Fold {fold} | Progresso: {i}/{len(imgs)}")

        original = cv2.imread(img_path)

        if original is None:
            print(f"⚠️ Erro ao ler: {img_path}")
            continue

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        h, w = original.shape[:2]

        img_tensor = preprocess_image(original_rgb).to(device)

        pred_mask = predict(model, img_tensor)

        # volta pro tamanho original
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # salvar máscara
        save_name = os.path.basename(img_path).rsplit('.', 1)[0] + ".png"

        pred_path = os.path.join(pred_fold_dir, save_name)
        cv2.imwrite(pred_path, pred_mask_resized.astype(np.uint8))

        # criar overlay
        overlay = create_overlay(original, pred_mask_resized)
        overlay_path = os.path.join(overlay_fold_dir, save_name)
        cv2.imwrite(overlay_path, overlay)

        if i % 50 == 0:
            print(f"💾 Salvo: {save_name}")

    end_fold = time.time()
    print(f"\n✅ Fold {fold} finalizado em {end_fold - start_fold:.2f}s")

print("\n🏁 Inferência concluída.")