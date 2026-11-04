from time import perf_counter
from threading import Lock
import cv2
from app.services.gap_detector import GapDetector
from app.services.product_detector import ProductDetector
from app.services.fusion import fuse


class Analyzer:
    def __init__(self, settings, product_detector=None, classical_only=False):
        self.settings = settings
        cv2.setNumThreads(settings.cpu_threads)
        self.inference_lock = Lock()
        self.gaps = GapDetector(settings)
        self.products = product_detector if product_detector is not None else ProductDetector(settings, not classical_only)

    def analyze(self, image):
        start = perf_counter()
        h, w = image.shape[:2]
        scale = min(1.0, self.settings.max_image_side / max(h, w))
        if scale < 1:
            image = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        raw = self.gaps.detect(image)
        with self.inference_lock:
            detections = self.products.predict(image)
            hybrid = self.products.available
            fallback_reason = self.products.reason
        fused = fuse(raw, detections, self.settings) if hybrid else raw
        annotated = self.gaps.annotate(image, fused)
        for item in detections:
            x1, y1, x2, y2 = map(round, item['bbox'])
            cv2.rectangle(annotated, (x1, y1), (x2-1, y2-1), (170, 200, 30), 2)
        result = {'mode': 'hybrid' if hybrid else 'classical_fallback', 'yolo_available': hybrid,
                  'fallback_reason': None if hybrid else fallback_reason,
                  'products_detected': len(detections) if hybrid else None,
                  'raw_gap_count': len(raw.regions), **fused.summary(), 'detections': detections,
                  'raw_classical': raw.summary(), 'fused': fused.summary(),
                  'original_size': [w, h], 'processed_size': [image.shape[1], image.shape[0]],
                  'coordinate_space': 'processed_image_xyxy_exclusive',
                  'processing_time_ms': round((perf_counter()-start)*1000, 3)}
        return result, annotated, raw.mask, fused.mask
