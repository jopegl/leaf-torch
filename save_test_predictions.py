import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pandas as pd

from modeling.deeplab_seg import DeepLab
from dataloaders.datasets.multi_leaf import MultiLeafDataset 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "area_lab/crossval_models_2_classes"
OUTPUT_DIR = "2_classes_test_images"
REPORT_DIR = "error_reports"

IMG_SIZE = 512
N_FOLDS = 5

MODE = "save"   # "error" ou "save"
LOW_IOU_THRESHOLD = 0.65
SAVE_CSV_REPORT = True

# =========================
# CONFIG DINÂMICA
# =========================
NUM_CLASSES = 2   # <-- troque para 2 ou 3 conforme seu modelo

CLASS_NAMES = {
    0: "background",
    1: "leaf",
    2: "pattern"
}

# Cores dinâmicas
COLORS = np.array([
    [0, 0, 0],        # background
    [0, 255, 0],      # leaf
    [255, 0, 0],      # pattern
], dtype=np.uint8)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(checkpoint_path, num_classes):
    model = DeepLab(
        num_classes=num_classes,
        backbone="xception",
        output_stride=16
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)

    model.to(DEVICE).eval()
    return model


def convert_mask_dynamic(mask_np, num_classes):
    """
    Converte a GT para o formato esperado pelo modelo.
    
    Regras:
    - num_classes = 2:
        0 = background
        1 = tudo que for folha/padrão (1 ou 2)
    - num_classes = 3:
        0 = background
        1 = leaf
        2 = pattern
    """
    if num_classes == 2:
        return ((mask_np == 1) | (mask_np == 2)).astype(np.uint8)

    elif num_classes == 3:
        gt = np.zeros_like(mask_np, dtype=np.uint8)
        gt[mask_np == 1] = 1
        gt[mask_np == 2] = 2
        return gt

    else:
        raise ValueError(f"NUM_CLASSES={num_classes} não suportado ainda.")


def compute_multiclass_ious(pred, target, num_classes):
    """
    Calcula IoU por classe + mIoU.
    """
    ious = {}
    valid_ious = []

    for cls in range(num_classes):
        pred_c = (pred == cls)
        target_c = (target == cls)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union

        ious[cls] = iou

        if not np.isnan(iou):
            valid_ious.append(iou)

    miou = np.mean(valid_ious) if len(valid_ious) > 0 else np.nan
    return ious, miou


def run_inference_and_metrics(model, image_path, mask_path, num_classes):
    # Imagem original
    img_raw = Image.open(image_path).convert("RGB")

    # Entrada do modelo
    input_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)

    # Ground truth mask
    gt_mask = Image.open(mask_path).convert("L")
    gt_mask = gt_mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    gt_mask = np.array(gt_mask)
    gt_mask = convert_mask_dynamic(gt_mask, num_classes)

    # Predição
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    ious, miou = compute_multiclass_ious(pred, gt_mask, num_classes)

    return pred, gt_mask, ious, miou


def save_prediction_figure(image_path, pred, save_path, num_classes):
    img_raw = Image.open(image_path).convert("RGB")
    img_np = np.array(img_raw.resize((IMG_SIZE, IMG_SIZE)))

    colors_used = COLORS[:num_classes]
    mask_colored = colors_used[pred].astype(np.uint8)
    overlay = (0.5 * img_np + 0.5 * mask_colored).astype(np.uint8)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("1. Imagem Original", fontsize=14, pad=10)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_colored)
    plt.title("2. Segmentação (Predição)", fontsize=14, pad=10)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("3. Sobreposição (Overlay)", fontsize=14, pad=10)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    all_results = []

    for fold in range(1, N_FOLDS + 1):
        print("\n" + "=" * 50)
        print(f"PROCESSANDO FOLD {fold}")
        print("=" * 50)

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"[AVISO] Checkpoint não encontrado: {checkpoint_path}")
            continue

        model = load_model(checkpoint_path, NUM_CLASSES)

        test_dataset = MultiLeafDataset(
            split="test",
            fold=fold,
            val_mode=False,
            num_classes=NUM_CLASSES
        )

        if MODE == "save":
            fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
            os.makedirs(fold_output_dir, exist_ok=True)

        fold_results = []

        print(f"Total de imagens de teste: {len(test_dataset.img_paths)}")

        for i, (img_path, mask_path) in enumerate(zip(test_dataset.img_paths, test_dataset.mask_paths), 1):
            filename = os.path.basename(img_path)
            image_name = os.path.splitext(filename)[0]

            try:
                pred, gt_mask, ious, miou = run_inference_and_metrics(
                    model, img_path, mask_path, NUM_CLASSES
                )

                result = {
                    "fold": fold,
                    "filename": filename,
                    "miou": miou
                }

                # adiciona iou por classe dinamicamente
                for cls in range(NUM_CLASSES):
                    class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                    result[f"iou_{class_name}"] = ious.get(cls, np.nan)

                fold_results.append(result)
                all_results.append(result)

                if MODE == "save":
                    save_path = os.path.join(fold_output_dir, f"{image_name}_pred.png")
                    save_prediction_figure(img_path, pred, save_path, NUM_CLASSES)
                    print(f"[{i}/{len(test_dataset.img_paths)}] Salvo: {save_path}")
                else:
                    msg = f"[{i}/{len(test_dataset.img_paths)}] {filename} | mIoU={miou:.4f}"
                    for cls in range(NUM_CLASSES):
                        class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                        cls_iou = ious.get(cls, np.nan)
                        msg += f" | IoU_{class_name}={cls_iou:.4f}" if not np.isnan(cls_iou) else f" | IoU_{class_name}=NaN"
                    print(msg)

            except Exception as e:
                print(f"[ERRO] Falha em {img_path}: {e}")

        if len(fold_results) > 0:
            fold_df = pd.DataFrame(fold_results)

            # métrica principal de suspeita = leaf
            leaf_col = "iou_leaf"
            if leaf_col not in fold_df.columns:
                print(f"[AVISO] Coluna {leaf_col} não encontrada. Pulando análise de suspeitas.")
                continue

            fold_mean_leaf = fold_df[leaf_col].mean()
            fold_mean_miou = fold_df["miou"].mean()
            fold_std_leaf = fold_df[leaf_col].std()

            outlier_threshold = (
                fold_mean_leaf - 2 * fold_std_leaf
                if pd.notna(fold_std_leaf)
                else LOW_IOU_THRESHOLD
            )

            suspicious_df = fold_df[
                (fold_df[leaf_col] < LOW_IOU_THRESHOLD) |
                (fold_df[leaf_col] < outlier_threshold)
            ].copy()

            suspicious_df = suspicious_df.sort_values(leaf_col)

            print("\n" + "-" * 50)
            print(f"RESUMO DO FOLD {fold}")
            print("-" * 50)
            print(f"Média IoU_leaf: {fold_mean_leaf:.4f}")
            print(f"Média mIoU: {fold_mean_miou:.4f}")

            # mostra médias por classe
            for cls in range(NUM_CLASSES):
                class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                col = f"iou_{class_name}"
                if col in fold_df.columns:
                    print(f"Média IoU_{class_name}: {fold_df[col].mean():.4f}")

            print(f"Desvio padrão IoU_leaf: {fold_std_leaf:.4f}" if pd.notna(fold_std_leaf) else "Desvio padrão IoU_leaf: NaN")
            print(f"Limiar absoluto: {LOW_IOU_THRESHOLD:.2f}")
            print(f"Limiar estatístico do fold: {outlier_threshold:.4f}")

            if len(suspicious_df) > 0:
                print(f"\n⚠️ Imagens suspeitas no fold {fold}:")
                for _, row in suspicious_df.iterrows():
                    print(
                        f" - {row['filename']} | "
                        f"IoU_leaf={row['iou_leaf']:.4f} | "
                        f"mIoU={row['miou']:.4f}"
                    )
            else:
                print("\nNenhuma imagem suspeita detectada neste fold.")

            if SAVE_CSV_REPORT:
                fold_csv = os.path.join(REPORT_DIR, f"fold_{fold}_metrics.csv")
                fold_df.sort_values(leaf_col).to_csv(fold_csv, index=False)
                print(f"\n[CSV] Relatório salvo em: {fold_csv}")

    # =========================
    # RESUMO GLOBAL
    # =========================
    if len(all_results) > 0:
        all_df = pd.DataFrame(all_results)

        leaf_col = "iou_leaf"

        global_mean_leaf = all_df[leaf_col].mean()
        global_mean_miou = all_df["miou"].mean()
        global_std_leaf = all_df[leaf_col].std()

        global_outlier_threshold = (
            global_mean_leaf - 2 * global_std_leaf
            if pd.notna(global_std_leaf)
            else LOW_IOU_THRESHOLD
        )

        worst_cases = all_df[
            (all_df[leaf_col] < LOW_IOU_THRESHOLD) |
            (all_df[leaf_col] < global_outlier_threshold)
        ].copy()

        worst_cases = worst_cases.sort_values(leaf_col)

        print("\n" + "=" * 60)
        print("RESUMO GLOBAL")
        print("=" * 60)
        print(f"Média global IoU_leaf: {global_mean_leaf:.4f}")
        print(f"Média global mIoU: {global_mean_miou:.4f}")

        for cls in range(NUM_CLASSES):
            class_name = CLASS_NAMES.get(cls, f"class_{cls}")
            col = f"iou_{class_name}"
            if col in all_df.columns:
                print(f"Média global IoU_{class_name}: {all_df[col].mean():.4f}")

        print(f"Desvio padrão global IoU_leaf: {global_std_leaf:.4f}" if pd.notna(global_std_leaf) else "Desvio padrão global IoU_leaf: NaN")
        print(f"Limiar absoluto: {LOW_IOU_THRESHOLD:.2f}")
        print(f"Limiar estatístico global: {global_outlier_threshold:.4f}")

        if len(worst_cases) > 0:
            print("\n🚨 PIORES CASOS GLOBAIS:")
            for _, row in worst_cases.iterrows():
                print(
                    f" - Fold {row['fold']} | {row['filename']} | "
                    f"IoU_leaf={row['iou_leaf']:.4f} | "
                    f"mIoU={row['miou']:.4f}"
                )
        else:
            print("\nNenhum caso crítico global detectado.")

        if SAVE_CSV_REPORT:
            global_csv = os.path.join(REPORT_DIR, "all_folds_metrics.csv")
            all_df.sort_values(leaf_col).to_csv(global_csv, index=False)
            print(f"\n[CSV] Relatório global salvo em: {global_csv}")

    print("\nProcesso concluído.")


if __name__ == "__main__":
    main()