import numpy as np

path = "split_dataset/test/004/area/leaf_004_002_area.raw"

data = np.fromfile(path, dtype=np.uint8)
print("Tamanho total (bytes):", data.nbytes)
print("Primeiros 20 valores:", data[:20])
print("Máximo e mínimo:", data.min(), data.max())

# Testa reshape
if data.size == 1024 * 1024:
    img = data.reshape((1024, 1024))
    print("Formato confirmado: 1024x1024")
else:
    print("Formato inesperado:", data.shape)

path = "split_dataset/test/004/segmentation/leaf_004_001_segmentation.raw"

data = np.fromfile(path, dtype=np.uint8)
print("Tamanho total (bytes):", data.nbytes)
print("Primeiros 20 valores:", data[:20])
print("Máximo e mínimo:", data.min(), data.max())

# Testa reshape
if data.size == 1024 * 1024:
    img = data.reshape((1024, 1024))
    print("Formato confirmado: 1024x1024")
else:
    print("Formato inesperado:", data.shape)

