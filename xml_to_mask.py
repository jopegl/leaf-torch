import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import shutil

base_dataset_dir = "multileaf_dataset/multileaf_dataset"
output_base_dir = "dataset_consolidado"
output_images_dir = os.path.join(output_base_dir, "images")
output_masks_dir = os.path.join(output_base_dir, "masks")

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

print("=== INICIANDO CONVERSÃO E CONSOLIDAÇÃO ===")

def extract_polygon_coords_onevision(polygon_node, scale_factor):
    """
    Extrai coordenadas x1, y1, x2, y2... da tag <polygon>.
    Normalizadas pela ALTURA.
    """
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

def process_xml_to_mask(xml_path, image_dir):
    xml_filename = os.path.basename(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERRO] Falha ao ler XML {xml_filename}: {e}")
        return False

    image_name_node = root.find("image-name")
    if image_name_node is None or not image_name_node.text:
        print(f"[PULAR] Tag <image-name> não encontrada em {xml_filename}")
        return False
    
    image_filename = image_name_node.text.strip()
    

    original_image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(original_image_path):
        base, _ = os.path.splitext(image_filename)
        for ext in ['.jpg', '.png', '.jpeg']:
            if os.path.exists(os.path.join(image_dir, base + ext)):
                original_image_path = os.path.join(image_dir, base + ext)
                image_filename = base + ext
                break
        
        if not os.path.exists(original_image_path):
            print(f"[ERRO] Imagem não encontrada: {image_filename}")
            return False

    img = cv2.imread(original_image_path)
    if img is None:
        print(f"[ERRO] Não foi possível abrir a imagem: {original_image_path}")
        return False

    height, width = img.shape[:2]
    

    mask = np.zeros((height, width), dtype=np.uint8)


    objects_node = root.find("objects")
    if objects_node is None:
        print(f"[AVISO] Tag <objects> não encontrada em {xml_filename}")
        return False

    for obj in objects_node:
        tag = obj.tag.lower()
        polygon_node = obj.find("polygon")
        
        if polygon_node is None:
            continue

        coords = extract_polygon_coords_onevision(polygon_node, scale_factor=height)
        
        if coords.size > 0:
            if tag == "leaf":
                cv2.fillPoly(mask, coords, color=1)
            elif tag == "pattern":
                cv2.fillPoly(mask, coords, color=2)


    base_name = os.path.splitext(image_filename)[0]
    out_mask_path = os.path.join(output_masks_dir, f"{base_name}.png")
    out_image_path = os.path.join(output_images_dir, image_filename)

    cv2.imwrite(out_mask_path, mask)
    shutil.copy2(original_image_path, out_image_path)

    return True


total_success = 0

for specie_folder in sorted(os.listdir(base_dataset_dir)):
    specie_path = os.path.join(base_dataset_dir, specie_folder)
    

    if not os.path.isdir(specie_path):
        continue

    labels_dir = os.path.join(specie_path, "labels")
    images_dir = os.path.join(specie_path, "images")

    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        continue

    xml_files = [f for f in os.listdir(labels_dir) if f.endswith(".xml")]
    print(f"Processando {specie_folder}: {len(xml_files)} arquivos...")
    
    specie_success = 0
    for xf in xml_files:
        if process_xml_to_mask(os.path.join(labels_dir, xf), images_dir):
            specie_success += 1
            total_success += 1

print(f"\n=== FINALIZADO ===")
print(f"Total de {total_success} pares (imagem/máscara) salvos em '{output_base_dir}' com seus nomes originais intocados.")