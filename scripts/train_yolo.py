import argparse
import hashlib
import shutil
from pathlib import Path
from scripts.common import environment, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/sku110k/dataset.yaml')
    p.add_argument('--model', choices=['yolov8n.pt', 'yolov8s.pt'], default='yolov8n.pt')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--name', default='shelf_yolov8n')
    args = p.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    data = Path(args.data).resolve()
    if not data.is_file():
        p.error('Prepare the dataset first')
    config = {**vars(args), 'data': args.data, 'environment': environment(),
              'dataset_manifest_sha256': hashlib.sha256((data.parent/'manifest.json').read_bytes()).hexdigest()}
    save_json(f'results/training/{args.name}_config.json', config)
    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, seed=args.seed, workers=0, deterministic=True,
        project=str(Path('runs/train').resolve()), name=args.name, exist_ok=False, patience=15,
        plots=True, max_det=1000, cache=False, amp=args.device != 'cpu')
    best = Path(model.trainer.best)
    Path('models').mkdir(exist_ok=True)
    shutil.copy2(best, 'models/shelf_yolov8n.pt')
    shutil.copy2(best.parent.parent/'results.csv', f'results/training/{args.name}_epochs.csv')
    import yaml
    training_args = yaml.safe_load((best.parent.parent/'args.yaml').read_text())
    for key in ('data', 'project', 'save_dir'):
        if key in training_args:
            try:
                training_args[key] = Path(training_args[key]).relative_to(Path.cwd()).as_posix()
            except ValueError:
                training_args[key] = Path(training_args[key]).name
    save_json(f'results/training/{args.name}_resolved_arguments.json', training_args)
    save_json(f'results/training/{args.name}_completion.json', {
        'best_weights_sha256': hashlib.sha256(best.read_bytes()).hexdigest(),
        'best_weights_local': 'models/shelf_yolov8n.pt', 'training_config': config})


if __name__ == '__main__':
    main()
