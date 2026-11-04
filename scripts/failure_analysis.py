"""Generate review artifacts; does NOT automatically score gap correctness."""
import argparse
import cv2
import numpy as np
from pathlib import Path
from app.api import decode_image
from app.config import Settings
from app.services.analyzer import Analyzer
from scripts.common import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, default=Path('data/sku110k/images/test'))
    p.add_argument('--output', type=Path, default=Path('results/failure_analysis'))
    p.add_argument('--limit', type=int, default=6)
    args = p.parse_args()
    s = Settings.from_env()
    analyzer = Analyzer(s)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))[:args.limit]
    if not paths:
        p.error('No images available')
    args.output.mkdir(parents=True, exist_ok=True)
    records, thumbnails = [], []
    for path in paths:
        image = decode_image(path.read_bytes(), s)
        result, annotated, raw, fused = analyzer.analyze(image)
        result['source_file'] = path.name
        save_json(args.output/f'{path.stem}.json', result)
        width, height = result['processed_size']
        original = cv2.resize(image, (width,height))
        classical = original.copy()
        active = raw > 0
        classical[active] = (0.6*classical[active]+0.4*np.array([30,175,245])).astype(np.uint8)
        panels = []
        for label, panel in [('Original',original),('Raw classical',classical),('Fused + products',annotated)]:
            canvas = np.full((400,400,3),245,np.uint8)
            factor = min(380/panel.shape[1],350/panel.shape[0])
            thumb = cv2.resize(panel,(round(panel.shape[1]*factor),round(panel.shape[0]*factor)))
            canvas[40:40+thumb.shape[0],10:10+thumb.shape[1]] = thumb
            cv2.putText(canvas, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX,.55,(40,60,40),1,cv2.LINE_AA)
            panels.append(canvas)
        combined = np.hstack(panels)
        cv2.imwrite(str(args.output/f'{path.stem}.jpg'), combined)
        cv2.imwrite(str(args.output/f'{path.stem}_raw.png'), raw)
        cv2.imwrite(str(args.output/f'{path.stem}_fused.png'), fused)
        thumbnails.append(combined)
        records.append({k: result[k] for k in ['source_file','mode','products_detected','raw_gap_count','gap_count','estimated_empty_area_percentage']})
    cv2.imwrite(str(args.output/'contact_sheet.jpg'), np.vstack(thumbnails))
    save_json(args.output/'index.json', {'note':'Qualitative review artifacts, no gap ground truth or accuracy score', 'images':records})
    print(records)


if __name__ == '__main__':
    main()
