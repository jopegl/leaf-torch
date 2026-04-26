import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF

from load_models import load_crossval_models
from modeling.deeplab_seg import DeepLab
from build_fold_image_index import build_fold_image_dict


# =========================
# CONFIG
# =========================

MODEL_PATHS = {
    1: os.path.join("crossval_models_2_classes", "best_model_fold_1.pth"),
    2: os.path.join("crossval_models_2_classes", "best_model_fold_2.pth"),
    3: os.path.join("crossval_models_2_classes", "best_model_fold_3.pth"),
    4: os.path.join("crossval_models_2_classes", "best_model_fold_4.pth"),
    5: os.path.join("crossval_models_2_classes", "best_model_fold_5.pth"),
}

CSV_PATH = "folds.csv"
DATASET_ROOT = "multileaf_dataset"
RESIZED_DATASET_PATH = "resized_multileaf/images"

DEVICE = "cpu"
IMG_SIZE = 512


# =========================
# MODEL
# =========================

def get_model():
    return DeepLab(
        num_classes=2,
        backbone="xception",
        output_stride=16,
        sync_bn=None,
        freeze_bn=True
    )


# =========================
# PREPROCESS
# =========================

def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)

    img = TF.normalize(
        img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return img.unsqueeze(0)


# =========================
# OVERLAY
# =========================

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    overlay = image.copy()

    colored = np.zeros_like(image)
    colored[mask == 1] = color

    overlay = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)

    return overlay


# =========================
# RUN
# =========================

def run():

    os.makedirs("testes_das_imagens", exist_ok=True)

    image_dict = build_fold_image_dict(CSV_PATH, DATASET_ROOT)
    models = load_crossval_models(get_model, DEVICE, MODEL_PATHS)

    for fold, model in models.items():

        print(f"\n==================== FOLD {fold} ====================\n")

        out_dir = os.path.join("testes_das_imagens", f"fold_{fold}")
        os.makedirs(out_dir, exist_ok=True)

        model.eval()

        count = 0

        for img_path in image_dict.get(fold, []):

            filename = os.path.basename(img_path)

            # =========================
            # ORIGINAL IMAGE
            # =========================
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                continue

            h, w = img_orig.shape[:2]

            # =========================
            # RESIZED IMAGE (FLAT DATASET)
            # =========================
            resized_path = os.path.join(RESIZED_DATASET_PATH, filename)
            img_resized = cv2.imread(resized_path)

            if img_resized is None:
                continue

            # =========================
            # PREDICTION
            # =========================
            inp = preprocess(img_resized).to(DEVICE)

            with torch.no_grad():
                out = model(inp)
                pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # =========================
            # RESIZE BACK TO ORIGINAL SIZE
            # =========================
            pred_up = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

            # =========================
            # OVERLAY
            # =========================
            overlay = overlay_mask(img_orig, pred_up)

            # =========================
            # SAVE
            # =========================
            save_path = os.path.join(out_dir, filename)
            cv2.imwrite(save_path, overlay)

            count += 1

            if count % 10 == 0:
                print(f"[Fold {fold}] Processadas {count} imagens")

        print(f"\n✔ Fold {fold} finalizado | total: {count} imagens")


if __name__ == "__main__":
    run()