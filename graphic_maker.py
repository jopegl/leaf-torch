import pandas as pd
import matplotlib.pyplot as plt
import os

# caminho do arquivo
csv_path = "resultados experimentos/poly_best_results/training_metrics.csv"

# pasta de saída
output_dir = "utils/graphics"
os.makedirs(output_dir, exist_ok=True)

# carregar dados
df = pd.read_csv(csv_path)

# métricas (uma por gráfico)
metrics = [
    "train_loss",
    "val_loss",
    "val_miou",
    "val_accuracy",
    "iou_background",
    "iou_leaf",
    "iou_marker"
]

# folds
folds = sorted(df["fold"].unique())

# gerar gráficos
for fold in folds:
    df_fold = df[df["fold"] == fold]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df_fold["epoch"], df_fold[metric])

        plt.title(f"Fold {fold} - {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid()

        # salvar imagem
        filename = f"fold_{fold}_{metric}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)

        plt.close()