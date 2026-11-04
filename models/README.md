# Models

The service expects `shelf_yolov8n.pt` (or `YOLO_MODEL_PATH`). Large weights are excluded from Git. It loads only local files at startup and never downloads a model implicitly.

`python -m scripts.train_yolo` explicitly starts from the official COCO-pretrained **YOLOv8n** model and fine-tunes one `product` class. COCO is initialization, not the deployed product detector. The script copies the validation-selected best checkpoint here and records its SHA-256 and training configuration under `results/training/`.

Only load trusted PyTorch checkpoints. Downloading arbitrary community weights is not required. Ultralytics software/weights licensing and dataset research-only terms remain applicable. See [research notes](../docs/research.md).

Missing, incompatible or failed weights activate classical fallback. A new checkpoint requires restarting the Flask process.
