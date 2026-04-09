import numpy as np
import torch
from utils.color import C

def compute_class_weights(self, dataset):
    """
    Calcula pesos inversamente proporcionais à frequência das classes.
    """
    class_counts = np.zeros(self.num_class, dtype=np.float64)

    for i in range(len(dataset)):
        _, mask, _, _, _, _, _ = dataset[i]
        for c in range(self.num_class):
            class_counts[c] += (mask == c).sum().item()

    total = class_counts.sum()
    freq = class_counts / total

    # Peso inverso à frequência
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * self.num_class  # normaliza para que a soma seja num_class

    print(f"{C.GREEN}✔ Calculated class weights:{C.END} {weights}")
    return torch.tensor(weights, dtype=torch.float32).cuda() if self.args.cuda else torch.tensor(weights, dtype=torch.float32)