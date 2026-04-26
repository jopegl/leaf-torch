import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

base_dataset_dir = "multileaf_dataset"

print("=== GERANDO MÁSCARAS NO DATASET ORIGINAL ===")


def extract_polygon_coords_onevision(polygon_node, scale_factor):
    coords = []
    i = 1

    while True:
        x_tag = polygon_node.find(f'x{i}')
        y_tag = polygon_node.find(f'y{i}')

        if x_tag is None or y_tag is None:
            break

        try:
            x_val = float(x_tag.text) * scale_factor
            y_val = float(y_tag.text) * scale_factor
            coords.append((int(x_val), int(y_val)))
        except (ValueError, TypeError):
            pass

        i += 1

    return np.array([coords], dtype=np.int32)


def process_xml(xml_path, images_dir, masks_dir):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERRO] XML inválido {xml_path}: {e}")
        return False

    image_name_node = root.find("image-name")
    if image_name_node is None or not image_name_node.text:
        return False

    image_filename = image_name_node.text.strip()
    image_path = os.path.join(images_dir, image_filename)

    if not os.path.exists(image_path):
        base, _ = os.path.splitext(image_filename)
        found = False
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(images_dir, base + ext)
            if os.path.exists(candidate):
                image_path = candidate
                image_filename = base + ext
                found = True
                break
        if not found:
            print(f"[ERRO] Imagem não encontrada: {image_filename}")
            return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    objects = root.find("objects")
    if objects is not None:
        for obj in objects:
            polygon = obj.find("polygon")
            if polygon is None:
                continue

            coords = extract_polygon_coords_onevision(polygon, scale_factor=h)

            if coords.size == 0:
                continue

            tag = obj.tag.lower()
            if tag == "leaf":
                cv2.fillPoly(mask, coords, 1)
            elif tag == "pattern":
                cv2.fillPoly(mask, coords, 2)

    base_name = os.path.splitext(image_filename)[0]
    mask_path = os.path.join(masks_dir, base_name + ".png")

    cv2.imwrite(mask_path, mask)
    return True


total = 0

for specie in sorted(os.listdir(base_dataset_dir)):
    specie_path = os.path.join(base_dataset_dir, specie)

    if not os.path.isdir(specie_path):
        continue

    images_dir = os.path.join(specie_path, "images")
    labels_dir = os.path.join(specie_path, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        continue

    # 👉 cria pasta masks dentro da própria espécie
    masks_dir = os.path.join(specie_path, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]

    print(f"\n[{specie}] processando {len(xml_files)} XMLs...")

    specie_count = 0

    for xml_file in xml_files:
        xml_path = os.path.join(labels_dir, xml_file)

        if process_xml(xml_path, images_dir, masks_dir):
            specie_count += 1
            total += 1

    print(f"[{specie}] concluído: {specie_count} máscaras")


print("\n=== FINALIZADO ===")
print(f"Total de máscaras geradas: {total}")