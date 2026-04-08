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
CHECKPOINT_DIR = "crossval_models"
OUTPUT_DIR = "test_images"
REPORT_DIR = "error_reports"

IMG_SIZE = 512
N_FOLDS = 5

MODE = "error"

LOW_IOU_THRESHOLD = 0.65

SAVE_CSV_REPORT = True

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

COLORS = np.array([
    [0, 0, 0],      # Fundo
    [0, 255, 0],    # Folha/objeto
], dtype=np.uint8)


def load_model(checkpoint_path):
    model = DeepLab(num_classes=2, backbone="xception", output_stride=16)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)

    model.to(DEVICE).eval()
    return model


def compute_binary_ious(pred, target):
    """
    pred, target: arrays 2D com valores 0 ou 1
    Retorna:
      - iou_bg
      - iou_leaf
      - miou_binary
    """
    ious = []

    for cls in [0, 1]:
        pred_c = (pred == cls)
        target_c = (target == cls)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union

        ious.append(iou)

    iou_bg, iou_leaf = ious

    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else np.nan

    return iou_bg, iou_leaf, miou


def run_inference_and_metrics(model, image_path, mask_path):
    # Imagem original
    img_raw = Image.open(image_path).convert("RGB")

    # Entrada do modelo
    input_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)

    # Ground truth mask
    gt_mask = Image.open(mask_path).convert("L")
    gt_mask = gt_mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    gt_mask = np.array(gt_mask)
    gt_mask = ((gt_mask == 1) | (gt_mask == 2)).astype(np.uint8)

    # Predição
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    iou_bg, iou_leaf, miou_binary = compute_binary_ious(pred, gt_mask)

    return pred, gt_mask, iou_bg, iou_leaf, miou_binary

def save_prediction_figure(image_path, pred, save_path):
    img_raw = Image.open(image_path).convert("RGB")
    img_np = np.array(img_raw.resize((IMG_SIZE, IMG_SIZE)))

    mask_colored = COLORS[pred].astype(np.uint8)
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

        model = load_model(checkpoint_path)

        test_dataset = MultiLeafDataset(
            split="test",
            fold=fold,
            val_mode=False
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
                pred, gt_mask, iou_bg, iou_leaf, miou_binary = run_inference_and_metrics(
                    model, img_path, mask_path
                )

                result = {
                    "fold": fold,
                    "filename": filename,
                    "iou_bg": iou_bg,
                    "iou_leaf": iou_leaf,
                    "miou_binary": miou_binary
                }

                fold_results.append(result)
                all_results.append(result)

                if MODE == "save":
                    save_path = os.path.join(fold_output_dir, f"{image_name}_pred.png")
                    save_prediction_figure(img_path, pred, save_path)
                    print(f"[{i}/{len(test_dataset.img_paths)}] Salvo: {save_path}")
                else:
                    print(
                        f"[{i}/{len(test_dataset.img_paths)}] {filename} | "
                        f"IoU_leaf={iou_leaf:.4f} | mIoU={miou_binary:.4f}"
                    )

            except Exception as e:
                print(f"[ERRO] Falha em {img_path}: {e}")

    
        if len(fold_results) > 0:
            fold_df = pd.DataFrame(fold_results)

            fold_mean_leaf = fold_df["iou_leaf"].mean()
            fold_mean_miou = fold_df["miou_binary"].mean()
            fold_std_leaf = fold_df["iou_leaf"].std()

            # Critério de outlier estatístico
            outlier_threshold = fold_mean_leaf - 2 * fold_std_leaf if pd.notna(fold_std_leaf) else LOW_IOU_THRESHOLD

            suspicious_df = fold_df[
                (fold_df["iou_leaf"] < LOW_IOU_THRESHOLD) |
                (fold_df["iou_leaf"] < outlier_threshold)
            ].copy()

            suspicious_df = suspicious_df.sort_values("iou_leaf")

            print("\n" + "-" * 50)
            print(f"RESUMO DO FOLD {fold}")
            print("-" * 50)
            print(f"Média IoU_leaf: {fold_mean_leaf:.4f}")
            print(f"Média mIoU binário: {fold_mean_miou:.4f}")
            print(f"Desvio padrão IoU_leaf: {fold_std_leaf:.4f}" if pd.notna(fold_std_leaf) else "Desvio padrão IoU_leaf: NaN")
            print(f"Limiar absoluto: {LOW_IOU_THRESHOLD:.2f}")
            print(f"Limiar estatístico do fold: {outlier_threshold:.4f}")

            if len(suspicious_df) > 0:
                print(f"\n⚠️ Imagens suspeitas no fold {fold}:")
                for _, row in suspicious_df.iterrows():
                    print(
                        f" - {row['filename']} | "
                        f"IoU_leaf={row['iou_leaf']:.4f} | "
                        f"mIoU={row['miou_binary']:.4f}"
                    )
            else:
                print("\nNenhuma imagem suspeita detectada neste fold.")

            if SAVE_CSV_REPORT:
                fold_csv = os.path.join(REPORT_DIR, f"fold_{fold}_metrics.csv")
                fold_df.sort_values("iou_leaf").to_csv(fold_csv, index=False)
                print(f"\n[CSV] Relatório salvo em: {fold_csv}")


    if len(all_results) > 0:
        all_df = pd.DataFrame(all_results)

        global_mean_leaf = all_df["iou_leaf"].mean()
        global_mean_miou = all_df["miou_binary"].mean()
        global_std_leaf = all_df["iou_leaf"].std()

        global_outlier_threshold = global_mean_leaf - 2 * global_std_leaf if pd.notna(global_std_leaf) else LOW_IOU_THRESHOLD

        worst_cases = all_df[
            (all_df["iou_leaf"] < LOW_IOU_THRESHOLD) |
            (all_df["iou_leaf"] < global_outlier_threshold)
        ].copy()

        worst_cases = worst_cases.sort_values("iou_leaf")

        print("\n" + "=" * 60)
        print("RESUMO GLOBAL")
        print("=" * 60)
        print(f"Média global IoU_leaf: {global_mean_leaf:.4f}")
        print(f"Média global mIoU binário: {global_mean_miou:.4f}")
        print(f"Desvio padrão global IoU_leaf: {global_std_leaf:.4f}" if pd.notna(global_std_leaf) else "Desvio padrão global IoU_leaf: NaN")
        print(f"Limiar absoluto: {LOW_IOU_THRESHOLD:.2f}")
        print(f"Limiar estatístico global: {global_outlier_threshold:.4f}")

        if len(worst_cases) > 0:
            print("\n🚨 PIORES CASOS GLOBAIS:")
            for _, row in worst_cases.iterrows():
                print(
                    f" - Fold {row['fold']} | {row['filename']} | "
                    f"IoU_leaf={row['iou_leaf']:.4f} | "
                    f"mIoU={row['miou_binary']:.4f}"
                )
        else:
            print("\nNenhum caso crítico global detectado.")

        if SAVE_CSV_REPORT:
            global_csv = os.path.join(REPORT_DIR, "all_folds_metrics.csv")
            all_df.sort_values("iou_leaf").to_csv(global_csv, index=False)
            print(f"\n[CSV] Relatório global salvo em: {global_csv}")

    print("\nProcesso concluído.")


if __name__ == "__main__":
    main()