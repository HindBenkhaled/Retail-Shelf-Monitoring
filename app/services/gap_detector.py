from dataclasses import dataclass
import cv2
import numpy as np
from app.config import Settings


@dataclass
class GapResult:
    mask: np.ndarray
    regions: list
    diagnostics: dict

    def summary(self):
        return {'gap_count': len(self.regions), 'gaps': self.regions,
                'estimated_empty_area_percentage': round(100 * np.count_nonzero(self.mask) / self.mask.size, 4),
                'gap_area_pixels': int(np.count_nonzero(self.mask)),
                'diagnostics': self.diagnostics}


def filter_regions(mask, settings):
    """Keep exact connected pixels (including holes), never rectangle area."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    lookup = np.zeros(count, np.uint8)
    regions = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area >= settings.min_gap_area and w >= settings.min_gap_width and h >= settings.min_gap_height:
            lookup[i] = 255
            regions.append({'bbox': [x, y, x+w, y+h], 'area_pixels': area})
    return lookup[labels], regions


class GapDetector:
    def __init__(self, settings=None):
        self.settings = settings or Settings()

    def detect(self, image):
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a nonempty uint8 BGR image')
        s = self.settings
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        energy = cv2.boxFilter(np.abs(laplacian), -1, (s.texture_window, s.texture_window))
        mask = (energy < s.laplacian_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s.morph_kernel, s.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask, regions = filter_regions(mask, s)
        return GapResult(mask, regions, {'mean_texture_energy': float(energy.mean()),
                         'laplacian_threshold': s.laplacian_threshold,
                         'denominator': 'entire_processed_image', 'heuristic': True})

    def annotate(self, image, result):
        annotated = image.copy()
        active = result.mask > 0
        annotated[active] = (0.6 * annotated[active] + 0.4 * np.array([30, 175, 245])).astype(np.uint8)
        for region in result.regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2-1, y2-1), (30, 175, 245), 2)
        return annotated
