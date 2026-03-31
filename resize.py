import os
import cv2

input_base_dir = "dataset_consolidado"
input_images_dir = os.path.join(input_base_dir, "images")
input_masks_dir = os.path.join(input_base_dir, "masks")

output_base_dir = "resized_multileaf"
output_images_dir = os.path.join(output_base_dir, "images")
output_masks_dir = os.path.join(output_base_dir, "masks")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

TARGET_SIZE = (512, 512)

print(f"=== INICIANDO REDIMENSIONAMENTO PARA {TARGET_SIZE} ===")

image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"Encontradas {len(image_files)} imagens para processar.")

success_count = 0

for img_filename in image_files:
    img_path = os.path.join(input_images_dir, img_filename)

    base_name = os.path.splitext(img_filename)[0]
    mask_filename = f"{base_name}.png"
    mask_path = os.path.join(input_masks_dir, mask_filename)

    if not os.path.exists(mask_path):
        print(f"[AVISO] Máscara não encontrada para a imagem: {img_filename}")
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"[ERRO] Falha ao ler o par: {img_filename}")
        continue

    resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    resized_mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    out_img_path = os.path.join(output_images_dir, img_filename)
    out_mask_path = os.path.join(output_masks_dir, mask_filename)

    cv2.imwrite(out_img_path, resized_img)
    cv2.imwrite(out_mask_path, resized_mask)

    success_count += 1

print("\n=== FINALIZADO ===")
print(f"Total de {success_count} pares (imagem/máscara) redimensionados e salvos em '{output_base_dir}'.")
print("Os arquivos na pasta 'dataset_consolidado' continuam intactos!")