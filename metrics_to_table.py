import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_metrics.csv')

df_fold1 = df[(df['fold'] == 1) & (df['epoch'] <= 70)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(df_fold1['epoch'], df_fold1['train_loss'], label='Train Loss', color='blue', lw=2)
ax1.plot(df_fold1['epoch'], df_fold1['val_loss'], label='Val Loss', color='red', linestyle='--', lw=2)
ax1.set_title('Histórico de Perda (Loss) - Fold 1', fontsize=14)
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df_fold1['epoch'], df_fold1['val_miou'], label='Val mIoU', color='green', lw=2)
ax2.plot(df_fold1['epoch'], df_fold1['val_accuracy'], label='Val Accuracy', color='orange', lw=2)
ax2.set_title('Métricas de Validação - Fold 1', fontsize=14)
ax2.set_xlabel('Época')
ax2.set_ylabel('Score (0 a 1)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()