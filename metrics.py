import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import csv

DATASET_ROOT = "multileaf_dataset"
DEBUG_DIR = "debug_masks"
os.makedirs(DEBUG_DIR, exist_ok=True)

print("=== PIPELINE COM MÉTRICAS POR FOLHA ===")


# =========================
# IOU
# =========================
def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-8)


# =========================
# MESMA LÓGICA DO SCRIPT ORIGINAL
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
        except (ValueError, TypeError):
            pass

        i += 1

    return np.array([coords], dtype=np.int32)


def polygon_to_mask(coords, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, coords, 1)
    return mask


# =========================
# PARSE XML
# =========================
def parse_xml(xml_path, img_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    h, w = img_shape[:2]

    leaves = []
    marker = None

    objects = root.find("objects")
    if objects is None:
        return None, None

    for obj in objects:
        polygon = obj.find("polygon")
        if polygon is None:
            continue

        coords = extract_polygon_coords_onevision(polygon, scale_factor=h)

        if coords.size == 0:
            continue

        if obj.tag.lower() == "leaf":
            leaves.append(polygon_to_mask(coords, (h, w)))

        elif obj.tag.lower() == "pattern":
            marker = polygon_to_mask(coords, (h, w))

    return leaves, marker


# =========================
# PRED MASK
# =========================
def load_pred(path, shape):
    m = cv2.imread(path, 0)
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


# =========================
# MATCHING
# =========================
def match(gt, pred):
    pairs = []

    for i, g in enumerate(gt):
        best_j = -1
        best = 0

        for j, p in enumerate(pred):
            score = iou(g, p)
            if score > best:
                best = score
                best_j = j

        pairs.append((i, best_j, best))

    return pairs


# =========================
# DEBUG VISUAL (DESATIVADO)
# =========================
def visualize_debug(img, gt_masks, pred_masks, save_path):

    vis_gt = img.copy()
    vis_pred = img.copy()
    vis_overlap = img.copy()

    for g in gt_masks:
        vis_gt[g > 0] = (0, 255, 0)

    for p in pred_masks:
        vis_pred[p > 0] = (255, 0, 0)

    for g in gt_masks:
        for p in pred_masks:
            vis_overlap[(g > 0) & (p > 0)] = (255, 255, 0)

    stacked = np.hstack([vis_gt, vis_pred, vis_overlap])
    cv2.imwrite(save_path, stacked)


# =========================
# MAIN
# =========================
results = []
records = []
leaf_counter = 0

for root, _, files in os.walk(DATASET_ROOT):
    for f in files:
        if not f.endswith(".jpg"):
            continue

        img_path = os.path.join(root, f)
        xml_path = img_path.replace("images", "labels").replace(".jpg", ".xml")
        pred_path = img_path.replace("images", "pred_masks").replace(".jpg", ".png")

        if not os.path.exists(xml_path) or not os.path.exists(pred_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        gt_leaves, marker = parse_xml(xml_path, (h, w))
        if gt_leaves is None or marker is None:
            continue

        pred = load_pred(pred_path, (h, w))
        pred_comps = split_components(pred)

        # =========================
        # DEBUG (comentado)
        # =========================
        # debug_path = os.path.join(
        #     DEBUG_DIR,
        #     os.path.basename(img_path).replace(".jpg", ".png")
        # )
        #
        # visualize_debug(img, gt_leaves, pred_comps, debug_path)
        # print(f"[DEBUG salvo] {debug_path}")

        # =========================
        # ESCALA
        # =========================
        MARKER_REAL_AREA_CM2 = 4.0
        pixel_area = marker.sum()
        scale = MARKER_REAL_AREA_CM2 / (pixel_area + 1e-8)

        # =========================
        # MATCH
        # =========================
        matches = match(gt_leaves, pred_comps)

        for i, j, score in matches:

            gt_area_px = gt_leaves[i].sum()
            gt_area_cm2 = gt_area_px * scale

            if j == -1:
                pred_area_px = 0
                pred_area_cm2 = 0
                mre = 1.0
            else:
                pred_area_px = pred_comps[j].sum()
                pred_area_cm2 = pred_area_px * scale
                mre = abs(pred_area_cm2 - gt_area_cm2) / (gt_area_cm2 + 1e-8)

            results.append(mre)

            records.append({
                "image": f,
                "leaf_index": i,
                "gt_area_cm2": gt_area_cm2,
                "pred_area_cm2": pred_area_cm2,
                "mre": mre
            })

            # =========================
            # PRINT A CADA 20 FOLHAS
            # =========================
            leaf_counter += 1

            if leaf_counter % 20 == 0:
                print(
                    f"[{leaf_counter} folhas] "
                    f"GT={gt_area_cm2:.3f} cm² | "
                    f"PRED={pred_area_cm2:.3f} cm² | "
                    f"MRE={mre:.4f}"
                )

        print(f"{f} | folhas={len(gt_leaves)} | scale={scale:.6f}")


# =========================
# CSV EXPORT
# =========================
csv_path = "leaf_metrics.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image",
        "leaf_index",
        "gt_area_cm2",
        "pred_area_cm2",
        "mre"
    ])

    writer.writeheader()
    writer.writerows(records)


# =========================
# FINAL RESULT
# =========================
print("\n=== RESULTADO FINAL ===")
print("MRE médio:", np.mean(results))
print("MRE std:", np.std(results))
print("Total folhas:", len(results))
print(f"CSV salvo em: {csv_path}")