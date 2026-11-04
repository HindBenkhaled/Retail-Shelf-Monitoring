import math
import numpy as np
from app.services.gap_detector import GapResult, filter_regions


def occupancy_mask(shape, detections, padding=0):
    mask = np.zeros(shape, np.uint8)
    h, w = shape
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        x1, y1 = max(0, math.floor(x1)-padding), max(0, math.floor(y1)-padding)
        x2, y2 = min(w, math.ceil(x2)+padding), min(h, math.ceil(y2)+padding)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def fuse(raw, detections, settings):
    occupied = occupancy_mask(raw.mask.shape, detections, settings.product_padding)
    mask = raw.mask.copy()
    mask[occupied > 0] = 0
    # Re-filter fragments, but do not close morphology and refill occupied pixels.
    mask, regions = filter_regions(mask, settings)
    return GapResult(mask, regions, {**raw.diagnostics,
        'suppressed_candidate_pixels': int(np.count_nonzero(raw.mask) - np.count_nonzero(mask)),
        'product_occupancy_pixels': int(np.count_nonzero(occupied))})
