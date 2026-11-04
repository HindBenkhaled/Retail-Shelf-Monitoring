import argparse
from pathlib import Path
from scripts.common import environment, save_json
from app.services.product_detector import validate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/sku110k/dataset.yaml')
    p.add_argument('--weights', default='models/shelf_yolov8n.pt')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--output', default='results/yolo_evaluation.json')
    args = p.parse_args()
    if not Path(args.weights).is_file() or not Path(args.data).is_file():
        p.error('Local weights and prepared data are required')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(args.weights)
    validate_model(model)
    metrics = model.val(data=str(Path(args.data).resolve()), split=args.split, imgsz=args.imgsz,
                        device=args.device, batch=args.batch, workers=0, max_det=1000, plots=True,
                        project=str(Path('runs/evaluation').resolve()), name=args.split)
    save_json(args.output, {'configuration': vars(args), 'environment': environment(),
        'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
        'note': 'Product bounding boxes only; no gap ground truth. Small subset evaluation.'})


if __name__ == '__main__':
    main()
