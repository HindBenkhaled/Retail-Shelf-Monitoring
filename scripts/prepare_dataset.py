"""Download a small SKU-110K mirror subset or convert official CSV annotations.

Original terms: academic/non-commercial use only. Images/labels remain ignored.
Mirror selection is a fixed prefix within each original split, not representative.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
from PIL import Image

REPO = 'benjamintli/sku110k'
REVISION = '41ffe587d4e568204bbf752ba8f18ea21e3d7b09'
SOURCE = 'https://github.com/eg4000/SKU110K_CVPR19'
TERMS = 'Exclusive use by recipient, solely academic and non-commercial purposes (original source).'


def yolo_box(box, width, height):
    x, y, w, h = map(float, box)
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid bounding box: {box}')
    return f'0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}'


def write_sample(root, split, name, content, boxes, expected_size=None):
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
        image.verify()
    if expected_size and (width, height) != expected_size:
        raise ValueError('Mirror thumbnail dimensions differ from annotation coordinate space')
    labels = [yolo_box(b, width, height) for b in boxes]
    images, annotations = root / 'images' / split, root / 'labels' / split
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    (images / name).write_bytes(content)
    (annotations / (Path(name).stem+'.txt')).write_text('\n'.join(labels)+'\n', encoding='utf-8')
    return {'split': split, 'file': name, 'sha256': hashlib.sha256(content).hexdigest(),
            'width': width, 'height': height, 'boxes': len(labels), 'bytes': len(content)}


def prepare_mirror(root, counts):
    samples = []
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    for split, remote_split in [('train', 'train'), ('val', 'validation'), ('test', 'test')]:
        for offset in range(counts[split]):
            existing = root/'images'/split/f'{split}_{offset:05d}.jpg'
            label = root/'labels'/split/f'{split}_{offset:05d}.txt'
            if existing.is_file() and label.is_file():
                content = existing.read_bytes()
                with Image.open(io.BytesIO(content)) as image:
                    width, height = image.size
                    image.verify()
                samples.append({'split': split, 'file': existing.name, 'sha256': hashlib.sha256(content).hexdigest(),
                    'width': width, 'height': height, 'boxes': len(label.read_text().splitlines()),
                    'bytes': len(content), 'source_row': offset})
                continue
            response = session.get('https://datasets-server.huggingface.co/rows', params={
                'dataset': REPO, 'config': 'default', 'split': remote_split,
                'offset': offset, 'length': 1}, timeout=30)
            response.raise_for_status()
            rows = response.json()['rows']
            if len(rows) != 1:
                raise ValueError('Incomplete mirror response')
            for item in rows:
                row, index = item['row'], item['row_idx']
                # Fail if the mirror changes; do not silently mix revisions.
                src = row['image']['src']
                if REVISION not in src:
                    raise ValueError('Mirror revision changed; review provenance before updating REVISION')
                if set(row['objects']['category']) != {0}:
                    raise ValueError('Unexpected class schema')
                name = f'{split}_{index:05d}.jpg'
                existing = root/'images'/split/name
                if existing.exists():
                    content = existing.read_bytes()
                else:
                    downloaded = session.get(src, timeout=60)
                    downloaded.raise_for_status()
                    content = downloaded.content
                sample = write_sample(root, split, name, content,
                    row['objects']['bbox'], (row['image']['width'], row['image']['height']))
                sample['source_row'] = index
                samples.append(sample)
            print(f'{split}: {offset+1}/{counts[split]}', flush=True)
    return samples


def prepare_official(root, source, counts):
    samples = []
    for split in ('train', 'val', 'test'):
        groups = {}
        with (source / 'annotations' / f'annotations_{split}.csv').open(newline='') as f:
            for name, x1, y1, x2, y2, category, width, height in csv.reader(f):
                if Path(name).name != name:
                    raise ValueError('Unsafe image filename')
                box = (float(x1), float(y1), float(x2)-float(x1), float(y2)-float(y1))
                groups.setdefault(name, []).append(box)
        for name in sorted(groups)[:counts[split]]:
            samples.append(write_sample(root, split, name, (source/'images'/name).read_bytes(), groups[name]))
        if sum(s['split'] == split for s in samples) != counts[split]:
            raise ValueError('Source has fewer images than requested')
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('data/sku110k'))
    p.add_argument('--official-root', type=Path)
    p.add_argument('--train', type=int, default=64)
    p.add_argument('--val', type=int, default=12)
    p.add_argument('--test', type=int, default=12)
    p.add_argument('--accept-research-terms', action='store_true', required=True)
    p.add_argument('--resume', action='store_true', help='Resume an interrupted download; complete manifests cannot be overwritten')
    args = p.parse_args()
    counts = {k: getattr(args, k) for k in ('train', 'val', 'test')}
    if min(counts.values()) < 1:
        p.error('Each split needs at least one image')
    if args.output.exists() and any(args.output.iterdir()) and (not args.resume or (args.output/'manifest.json').exists()):
        p.error('Output must be empty; choose a new output directory to preserve previous manifests')
    args.output.mkdir(parents=True, exist_ok=True)
    samples = prepare_official(args.output, args.official_root, counts) if args.official_root else prepare_mirror(args.output, counts)
    hashes = [s['sha256'] for s in samples]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Duplicate image bytes detected; review splits')
    manifest = {'name': 'SKU-110K', 'source': SOURCE, 'terms': TERMS,
        'mirror': None if args.official_root else f'https://huggingface.co/datasets/{REPO}',
        'revision': None if args.official_root else REVISION,
        'selection': 'fixed prefix per original split (mirror order) or sorted official filenames',
        'class_mapping': {'0': 'product'}, 'counts': counts, 'samples': samples,
        'gap_ground_truth': False, 'total_image_bytes': sum(s['bytes'] for s in samples)}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output/'dataset.yaml').write_text(yaml.safe_dump({'path': args.output.resolve().as_posix(),
        'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': {0: 'product'}}), encoding='utf-8')
    Path('results').mkdir(exist_ok=True)
    Path('results/dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'counts': counts, 'image_bytes': manifest['total_image_bytes']}))


if __name__ == '__main__':
    main()
