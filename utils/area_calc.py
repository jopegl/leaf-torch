import cv2
import numpy as np

def calculate_leaf_area(mask, pattern_side, orig_w=None, orig_h=None):
    mask = np.array(mask)

    if orig_w is not None and orig_h is not None:
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    leaf_mask = (mask == 1).astype(np.uint8)
    marker_mask = (mask == 2).astype(np.uint8)

    marker_area_pixels = np.count_nonzero(marker_mask)
    marker_area_cm2 = pattern_side ** 2

    if marker_area_pixels == 0:
        return {
            "conversion_factor": None,
            "marker_area_pixels": 0,
            "marker_area_cm2": marker_area_cm2,
            "total_leaf_area_pixels": 0,
            "total_leaf_area_cm2": None,
            "leafs": []
        }

    conversion_factor = marker_area_cm2 / marker_area_pixels

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leafs = []
    total_leaf_area_pixels = 0.0
    leaf_index = 1

    for contour in contours:
        area_pixels = cv2.contourArea(contour)

        if area_pixels < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        area_cm2 = area_pixels * conversion_factor

        leafs.append({
            "index": leaf_index,
            "area_pixels": area_pixels,
            "area_cm2": area_cm2,
            "bbox": (x, y, w, h),
            "contour": contour
        })

        total_leaf_area_pixels += area_pixels
        leaf_index += 1

    total_leaf_area_cm2 = total_leaf_area_pixels * conversion_factor

    return {
        "conversion_factor": conversion_factor,
        "marker_area_pixels": marker_area_pixels,
        "marker_area_cm2": marker_area_cm2,
        "total_leaf_area_pixels": total_leaf_area_pixels,
        "total_leaf_area_cm2": total_leaf_area_cm2,
        "leafs": leafs
    }