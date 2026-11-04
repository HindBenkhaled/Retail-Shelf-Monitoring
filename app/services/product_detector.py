"""Optional YOLOv8 adapter. Missing/recoverable failures latch fallback until restart."""
import logging
import pickle
from pathlib import Path
from threading import Lock
import numpy as np

log = logging.getLogger(__name__)
RECOVERABLE = (ImportError, OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError)


def validate_model(model):
    yaml = model.model.yaml
    architecture = str(yaml.get('yaml_file', ''))
    if not architecture and isinstance(getattr(model, 'ckpt', None), dict):
        architecture = str(model.ckpt.get('train_args', {}).get('model', ''))
    if not Path(architecture).name.lower().startswith('yolov8') or model.task != 'detect':
        raise ValueError('Only YOLOv8 detection models are supported')
    names = model.names
    if len(names) != 1 or str(names[0]).lower() not in ('product', 'object'):
        raise ValueError('Expected a single-class shelf product model; generic COCO weights are not accepted')


def adapt_boxes(result, width, height, confidence):
    detections = []
    if result.boxes is None:
        return detections
    for box, score, category in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                     result.boxes.cls.cpu().numpy()):
        if not np.isfinite(box).all() or not np.isfinite(score):
            raise ValueError('Non-finite detector output')
        if score < confidence or int(category) != 0:
            continue
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = np.clip([x1, x2], 0, width).tolist()
        y1, y2 = np.clip([y1, y2], 0, height).tolist()
        if x2 > x1 and y2 > y1:
            detections.append({'bbox': [x1, y1, x2, y2], 'confidence': float(score),
                               'class_id': 0, 'class_name': 'product'})
    return detections


class ProductDetector:
    def __init__(self, settings, enabled=True, loader=None):
        self.settings, self.model = settings, None
        self.reason = 'disabled' if not enabled else None
        self.lock = Lock()
        if not enabled:
            return
        if not Path(settings.yolo_model_path).is_file():
            self._disable('weights_missing')
            return
        try:
            if loader is None:
                import torch
                from ultralytics import YOLO
                torch.set_num_threads(settings.cpu_threads)
                loader = YOLO
            self.model = loader(settings.yolo_model_path)
            validate_model(self.model)
            log.info('YOLOv8 product model loaded')
        except RECOVERABLE:
            log.warning('YOLO loading failed; activating classical fallback', exc_info=True)
            self._disable('model_load_failed')

    @property
    def available(self):
        return self.model is not None

    def _disable(self, reason):
        self.model, self.reason = None, reason
        log.warning('Classical fallback active: %s', reason)

    def predict(self, image):
        with self.lock:
            if not self.available:
                return []
            try:
                result = self.model.predict(image, conf=self.settings.yolo_confidence,
                    imgsz=self.settings.yolo_image_size, device=self.settings.yolo_device,
                    max_det=1000, verbose=False)[0]
                return adapt_boxes(result, image.shape[1], image.shape[0], self.settings.yolo_confidence)
            except RECOVERABLE:
                log.warning('YOLO inference failed; activating classical fallback', exc_info=True)
                self._disable('inference_failed')
                return []
