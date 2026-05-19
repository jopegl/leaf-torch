import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import csv
import re
import pandas as pd
from pathlib import Path
from complete_csv import complete_csv
from adapt_to_plot import adapt_to_plot
DATASET_ROOT = "pos-correcoes/multileaf_dataset"

print("=== PIPELINE COM DETECÇÃO DE MARCADOR POR GEOMETRIA ===")


# =========================
# IOU
# =========================
def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)


# =========================
# XML PARSER (FOLHAS)
# =========================
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
        except:
            pass

        i += 1

    return np.array([coords], dtype=np.int32)

def safe_float(text):
    try:
        if text is None:
            return -1
        if str(text).strip().lower() in ["napp", "nan", "none", "null", ""]:
            return -1
        return float(text)
    except:
        return -1
    
def get_fold(f):
    file_name = Path(f).name
    
    try:

        df_folds = pd.read_csv("folds.csv")
        resultado = df_folds.loc[df_folds['Image Name'] == file_name, 'Fold']
        
        if not resultado.empty:
            return int(resultado.values[0])
        else:
            print(f"⚠️ Aviso: Arquivo {file_name} não encontrado no folds.csv")
            return 1 
            
    except Exception as e:
        print(f"❌ Erro ao ler folds.csv: {e}")
        return 1
    
def polygon_to_mask(coords, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, coords, 1)
    return mask

def debug_bad_marker(
    mask,
    image_name,
    chosen_cnt=None,
    bad_leaf_mask=None,
    save_dir="bad_marker_debug"
):
    os.makedirs(save_dir, exist_ok=True)

    vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        (_, _), (w, h), _ = rect

        if w <= 1 or h <= 1:
            continue

        aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
        rect_area = w * h
        fill_ratio = area / (rect_area + 1e-8)

        score = 0
        score += abs(aspect_ratio - 1.0) * 5
        score += abs(fill_ratio - 1.0) * 3
        score += 1 / (area + 1e-8)

        # candidatos em amarelo
        cv2.drawContours(vis, [box], 0, (0, 255, 255), 2)

        x, y = box[0]

        cv2.putText(
            vis,
            f"{score:.2f}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # marcador escolhido em verde
    if chosen_cnt is not None:
        rect = cv2.minAreaRect(chosen_cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.drawContours(vis, [box], 0, (0, 255, 0), 4)

    # =========================
    # DESTACA FOLHA COM RER ALTO
    # =========================
    if bad_leaf_mask is not None and bad_leaf_mask.sum() > 0:

        leaf_contours, _ = cv2.findContours(
            (bad_leaf_mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # contorno vermelho grosso
        cv2.drawContours(vis, leaf_contours, -1, (0, 0, 255), 4)

        # overlay semi-transparente
        overlay = vis.copy()

        cv2.drawContours(
            overlay,
            leaf_contours,
            -1,
            (0, 0, 255),
            -1
        )

        alpha = 0.25
        vis = cv2.addWeighted(
            overlay,
            alpha,
            vis,
            1 - alpha,
            0
        )

    save_path = os.path.join(
        save_dir,
        image_name.replace(".jpg", "_debug.png")
    )

    cv2.imwrite(save_path, vis)


def get_species(filename):
    name_only = os.path.splitext(filename)[0].lower()
    species = re.split(r'\d+', name_only)[0]
    
    return species


def parse_xml_leaves(xml_path, img_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    h, w = img_shape[:2]

    leaves = []

    objects = root.find("objects")
    if objects is None:
        return None

    for obj in objects:
        polygon = obj.find("polygon")
        perimeter = obj.find('dimensions/perimeter')
        area = obj.find('dimensions/area')
        length = obj.find('dimensions/length')
        if polygon is None:
            continue

        coords = extract_polygon_coords_onevision(polygon, scale_factor=h)

        if coords.size == 0:
            continue

        if obj.tag.lower() == "leaf":
            leaves.append({
                'mask':polygon_to_mask(coords, (h, w)),
                'perimeter': safe_float(perimeter.text if perimeter is not None else -1),
                'area':safe_float(area.text if area is not None else -1),
                'length': safe_float(length.text if length is not None else -1)
                })

    return leaves

def parse_xml_all(xml_path, img_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    objects = root.find("objects")

    for obj in objects:
        polygon = obj.find("polygon")
        if polygon is None:
            continue

        coords = extract_polygon_coords_onevision(polygon, scale_factor=h)

        if coords.size == 0:
            continue
        
        cv2.fillPoly(mask, coords, 1)
    
    return mask


def parse_xml_marker(xml_path, img_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    h, w = img_shape[:2]
    objects = root.find("objects")
    if objects is None:
        return None
    marker = {}
    for obj in objects:
        polygon = obj.find("polygon")
        if polygon is None:
            continue
        coords = extract_polygon_coords_onevision(polygon, scale_factor=h)
        if coords.size == 0:
            continue

        if obj.tag.lower() == "pattern":
            marker['mask'] = polygon_to_mask(coords, (h, w))
    return marker

def parse_xml_meta(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    capture_distance = root.find(".//capture-distance")
    pattern_side = root.find(".//pattern-side")

    return {
        "capture_distance": safe_float(capture_distance.text if capture_distance is not None else -1),
        "pattern_side": safe_float(pattern_side.text if pattern_side is not None else -1),
    }




def mask_perimeter(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0

    cnt = max(contours, key=cv2.contourArea)
    return cv2.arcLength(cnt, True)


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return 0, 0

    width = xs.max() - xs.min()
    height = ys.max() - ys.min()

    return max(height, width), min(height, width)


# =========================
# PREDIÇÃO
# =========================
def load_pred(path, shape):
    m = cv2.imread(path, 0)
    if m is None:
        return None

    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def split_components(mask):
    num, labels = cv2.connectedComponents(mask)
    comps = []

    for i in range(1, num):
        c = (labels == i).astype(np.uint8)
        if c.sum() > 50:
            comps.append(c)

    return comps

def debug_marker_detection(mask, image_name, save_dir="debug_markers"):
    os.makedirs(save_dir, exist_ok=True)

    vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
    

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # desenha contorno original (azul)
        cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 1)

        # desenha aproximação (verde ou vermelho)
        if len(approx) == 4:
            if is_valid_quadrilateral(approx, cos_threshold=0.6):
                color = (0, 255, 0)  # bom candidato verde
            else:
                color = (0, 0, 255)  # rejeitado pelos ângulos vermelho
        else:
            color = (0, 165, 255)  # não é quadrilátero laranja

        cv2.drawContours(vis, [approx], -1, color, 2)

    save_path = os.path.join(save_dir, image_name.replace(".jpg", "_debug.png"))
    cv2.imwrite(save_path, vis)

def estimate_length_px(mask):

    if mask is None or mask.sum() == 0:
        return 0.0

    mask_u8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)

    # fallback simples (contorno ruim)
    if len(cnt) < 5:
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        return float(max(w, h))

    # PCA no contorno
    pts = cnt.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    proj = cv2.PCAProject(pts, mean, eigenvectors)

    length = float(np.max(proj) - np.min(proj))

    return length

# =========================
# DETECÇÃO DO MARCADOR
# =========================

def angle_cos(p0, p1, p2):
    d1 = p0 - p1
    d2 = p2 - p1
    return np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8)

def is_valid_quadrilateral(approx, cos_threshold=0.6):
    if len(approx) != 4:
        return False

    approx = approx.reshape(4, 2)
    for i in range(4):
        p0 = approx[i]
        p1 = approx[(i + 1) % 4]
        p2 = approx[(i + 2) % 4]

        cos_angle = angle_cos(p0, p1, p2)
        if abs(cos_angle) > cos_threshold:
            return False
    return True

def find_marker_by_geometry(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_candidate = None
    best_score = float("inf")

    MIN_AREA = 500

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # remove ruídos pequenos
        if area < MIN_AREA:
            continue

        # retângulo mínimo do contorno
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect

        if w <= 1 or h <= 1:
            continue

        # quão próximo de quadrado
        aspect_ratio = max(w, h) / (min(w, h) + 1e-8)

        # quanto o contorno preenche o retângulo
        rect_area = w * h
        fill_ratio = area / (rect_area + 1e-8)

        # score menor = melhor candidato
        score = 0

        # quadrado ideal -> ratio = 1
        score += abs(aspect_ratio - 1.0) * 5

        # marcador deve preencher bem o retângulo
        score += abs(fill_ratio - 1.0) * 3

        # favorece objetos maiores
        score += 1 / (area + 1e-8)

        if score < best_score:
            best_score = score
            best_candidate = cnt

    if best_candidate is None:
        return None, None

    marker_mask = np.zeros(mask.shape, dtype=np.uint8)

    cv2.drawContours(
        marker_mask,
        [best_candidate],
        -1,
        1,
        -1
    )

    return marker_mask, best_candidate

# =========================
# MATCHING
# =========================
def match(gt, pred):
    pairs = []

    for i, g in enumerate(gt):
        best_j = -1
        best = 0

        for j, p in enumerate(pred):
            score = iou(g['mask'], p)
            if score > best:
                best = score
                best_j = j

        pairs.append((i, best_j, best))

    return pairs


def get_marker_side_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    node = root.find(".//pattern-side")
    if node is None:
        return None

    return float(node.text)


# =========================
# MAIN
# =========================
results = []
records = []
leaf_counter = 0


def find_pred_mask(img_path, preds_root):
    filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"

    for root, _, files in os.walk(preds_root):
        if filename in files:
            return os.path.join(root, filename)

    return None


for root, _, files in os.walk(DATASET_ROOT):
    for num, f in enumerate(files):
        
            if not f.endswith(".jpg"):
                continue
            
            img_path = os.path.join(root, f)
            xml_path = img_path.replace("images", "labels").replace(".jpg", ".xml")
            pred_path = find_pred_mask(img_path, 'preds')

            if not os.path.exists(xml_path) or not os.path.exists(pred_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            meta = parse_xml_meta(xml_path)

            gt_leaves = parse_xml_leaves(xml_path, (h, w))
            if gt_leaves is None:
                continue

            pred = load_pred(pred_path, (h, w))
            annot = parse_xml_all(xml_path, (h,w))
            
            if pred is None:
                continue
            
            gt_marker = parse_xml_marker(xml_path, (h,w))

            if gt_marker is None:
                continue

            marker, marker_cnt = find_marker_by_geometry(pred)
            annot_marker, annot_cnt = find_marker_by_geometry(annot)
  

            if marker is None or marker.sum() == 0:
                print(f"⚠️ Marker não encontrado: {f}")
                #debug_marker_detection(pred, f)
                continue
            if annot_marker is None or annot_marker.sum() == 0:
                print(f"⚠️ Annot Marker não encontrado: {f}")
                #debug_marker_detection(annot, f, save_dir="annot_debug_markers")
                continue

            pred_comps = split_components(pred)
            annot_comps = split_components(annot)

            # =========================i
            # ESCALA (CORRIGIDA)
            # =========================
            side = get_marker_side_from_xml(xml_path)
            marker_real_area = side * side

            pixel_area = cv2.contourArea(marker_cnt)
            pixel_annot_area = cv2.contourArea(annot_cnt)

            scale = marker_real_area / (pixel_area + 1e-8)
            annot_scale = marker_real_area/(pixel_annot_area + 1e-8)

            cm_per_pixel = np.sqrt(scale)
            annot_cm_per_pixel = np.sqrt(annot_scale)

            matches = match(gt_leaves, pred_comps)
            annot_matches = match(gt_leaves, annot_comps)

            for i, j, score in matches:
                
                gt_mask = gt_leaves[i]['mask']
                pred_mask = pred_comps[j] if j != -1 else np.zeros_like(gt_mask)

                pred_area_px = pred_mask.sum()
                
                gt_area_cm2 = float(gt_leaves[i]['area'])
                pred_area_cm2 = pred_area_px * scale
                annot_area_cm2 = gt_mask.sum() * annot_scale

                gt_perim_cm = float(gt_leaves[i]['perimeter'])
                pred_perim_cm = mask_perimeter(pred_mask) * cm_per_pixel
                annot_perim = mask_perimeter(gt_mask) * annot_cm_per_pixel

                pred_length_cm = estimate_length_px(pred_mask) * cm_per_pixel
                gt_length_cm = gt_leaves[i]['length']
                annot_length = estimate_length_px(gt_mask) * annot_cm_per_pixel

                if(gt_area_cm2 != -1):
                    area_rer = abs(pred_area_cm2 - gt_area_cm2) / (gt_area_cm2 + 1e-8)
                    if area_rer > 1:
                        print(f"🚨 RER alto detectado: {f}")
                        print(i, area_rer)

                        debug_bad_marker(
                            pred,
                            f,
                            bad_leaf_mask=pred_mask,
                            chosen_cnt=marker_cnt,
                            save_dir="high_rer_debug"
                        )
                    gt_area_rer = abs(annot_area_cm2 - gt_area_cm2)/(gt_area_cm2 + 1e-8)
                else:
                    area_rer = np.nan
                    gt_area_rer = np.nan
                if (gt_perim_cm != -1): 
                    perim_rer = abs(pred_perim_cm - gt_perim_cm) / (gt_perim_cm + 1e-8)
                    gt_perim_rer = abs(annot_perim - gt_perim_cm) / (gt_perim_cm + 1e-8)
                else:
                    perim_rer = np.nan
                    gt_perim_rer = np.nan

                if(gt_length_cm != -1):
                    length_rer = abs(pred_length_cm - gt_length_cm) / (gt_length_cm + 1e-8)
                    gt_length_rer = abs(annot_length - gt_length_cm) / (gt_length_cm + 1e-8)
                else:
                    length_rer = np.nan
                    gt_length_rer = np.nan

                if j == -1:
                    mre = 1.0
                else:
                    mre = abs(pred_area_cm2 - gt_area_cm2) / (gt_area_cm2 + 1e-8)
                
                area_method_rer = abs(pred_area_cm2 - annot_area_cm2) / (annot_area_cm2 + 1e-8)
                perim_method_rer = abs(pred_perim_cm - annot_perim) / (annot_perim + 1e-8)
                length_method_rer = abs(pred_length_cm - annot_length) / (annot_length + 1e-8)

                specie = get_species(f)
                fold = get_fold(f)

                results.append(mre)

                if num % 50 == 0:
                    print("\n" + "="*60)
                    print(f"IMAGE {num}: {f} | LEAF: {i}")
                    print("-"*60)

                    print(f"AREA:")
                    print(f"  GT   = {gt_area_cm2} cm²")
                    print(f"  PRED = {pred_area_cm2} cm²")
                    print(f"  RER  = {area_rer}")
                    print(f"  GT_ANNOT = {annot_area_cm2:.3f} cm² | RER_ANNOT = {gt_area_rer:.4f}")

                    print(f"\nPERIMETER:")
                    print(f"  GT   = {gt_perim_cm} cm")
                    print(f"  PRED = {pred_perim_cm} cm")
                    print(f"  RER  = {perim_rer}")
                    print(f"  GT_ANNOT = {annot_perim:.3f} cm | RER_ANNOT = {gt_perim_rer:.4f}")


                    print(f"\nLENGTH:")
                    print(f"  GT   = {gt_length_cm} cm")
                    print(f"  PRED = {pred_length_cm} cm")
                    print(f"  RER  = {length_rer}")
                    print(f"  GT_ANNOT = {annot_length:.3f} cm | RER_ANNOT = {gt_length_rer:.4f}")

                    print(f"\nMETA:")
                    print(f"  capture_distance = {meta['capture_distance']}")
                    print(f"  pattern_side     = {meta['pattern_side']}")
                    print(f"  specie          = {specie}")
                    print(f"  fold            = {fold}")

                    print('Methods RER:')
                    print(f"  Area method RER = {area_method_rer:.4f}")
                    print(f"  Perim method RER = {perim_method_rer:.4f}")
                    print(f"  Length method RER = {length_method_rer:.4f}")
                    print('Specie:', specie)


                    print("="*60 + "\n")
                records.append({
                    "image_name": f,
                    "leaf_index": i,
                    'fold':fold,
                    'species': specie,

                    "real_area_cm2": gt_area_cm2,
                    "pred_area_cm2": pred_area_cm2,
                    "area_rer": area_rer,
                    'area_annot': annot_area_cm2,
                    'area_annot_rer': gt_area_rer,

                    "real_perimeter_cm": gt_perim_cm,
                    "pred_perimeter_cm": pred_perim_cm,
                    "pred_perimeter_rer": perim_rer,
                    'perim_annot': annot_perim,
                    'perim_annot_rer': gt_perim_rer,
          
                    "real_length_cm": gt_length_cm,
                    "pred_length_cm": pred_length_cm,
                    "length_rer": length_rer,
                    'length_annot': annot_length,
                    'length_annot_rer': gt_length_rer,

                    "dist": meta["capture_distance"],
                    "pattern-size": meta['pattern_side'],
  

                    'area_method_rer': area_method_rer,
                    'perim_method_rer': perim_method_rer,
                    'length_method_rer': length_method_rer
    
                })

     



arr = np.array(results)

print("\n=== RESULTADOS ===")
print("MRE médio:", np.mean(arr))
print("MRE std:", np.std(arr))
print("Total folhas:", len(arr))


# =========================
# CSV FINAL
# =========================
csv_path = "FINAL_all_leaf_metrics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
    "image_name",
    "leaf_index",
    "fold",
    'species',

    "real_area_cm2",
    "pred_area_cm2",
    "area_rer",
    "area_annot",
    "area_annot_rer",

    "real_perimeter_cm",
    "pred_perimeter_cm",
    "pred_perimeter_rer",
    "perim_annot",
    "perim_annot_rer",

    "real_length_cm",
    "pred_length_cm",
    "length_rer",
    "length_annot",
    "length_annot_rer",

    "dist",
    "pattern-size",


    "area_method_rer",
    "perim_method_rer",
    "length_method_rer"
])

    writer.writeheader()
    writer.writerows(records)

print("\nCSV salvo em:", csv_path)

complete_csv()
adapt_to_plot() #analisar corn200