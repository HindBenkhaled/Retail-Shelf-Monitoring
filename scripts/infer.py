import argparse
import cv2
from app.config import Settings
from app.api import decode_image
from app.services.analyzer import Analyzer
from scripts.common import save_json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image', type=Path)
    p.add_argument('--classical-only', action='store_true')
    p.add_argument('--output', type=Path, default=Path('outputs/inference'))
    args = p.parse_args()
    s = Settings.from_env()
    image = decode_image(args.image.read_bytes(), s)
    result, annotated, raw, fused = Analyzer(s, classical_only=args.classical_only).analyze(image)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/'analysis.json', result)
    for name, image in [('annotated', annotated), ('raw_mask', raw), ('fused_mask', fused)]:
        if not cv2.imwrite(str(args.output/f'{name}.png'), image):
            raise OSError('Image write failed')
    print(result)


if __name__ == '__main__':
    main()
