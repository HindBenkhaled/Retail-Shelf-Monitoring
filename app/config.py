"""Validated, centralized configuration; pixel thresholds apply after resizing."""
import os
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    yolo_model_path: str = 'models/shelf_yolov8n.pt'
    yolo_confidence: float = 0.25
    yolo_image_size: int = 416
    yolo_device: str = 'cpu'
    cpu_threads: int = 4
    max_image_side: int = 1280
    min_gap_area: int = 600
    min_gap_width: int = 15
    min_gap_height: int = 15
    laplacian_threshold: float = 12.0
    texture_window: int = 15
    morph_kernel: int = 5
    product_padding: int = 3
    max_upload_mb: int = 10
    max_image_pixels: int = 20_000_000
    output_dir: str = 'outputs'
    result_ttl_seconds: int = 3600
    max_results: int = 100

    def __post_init__(self):
        for name in ('texture_window', 'morph_kernel'):
            value = getattr(self, name)
            if value < 1 or value % 2 == 0:
                raise ValueError(f'{name} must be a positive odd integer')
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f'{f.name} cannot be negative')
        for name in ('max_image_side', 'max_upload_mb', 'max_image_pixels', 'max_results', 'yolo_image_size', 'result_ttl_seconds', 'cpu_threads'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if not 0 < self.yolo_confidence <= 1:
            raise ValueError('yolo_confidence must be in (0, 1]')

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / '.env')
        return cls(**{f.name: f.type(os.environ[f.name.upper()]) for f in fields(cls)
                      if f.name.upper() in os.environ})

    @property
    def output_path(self):
        return Path(self.output_dir).resolve()
