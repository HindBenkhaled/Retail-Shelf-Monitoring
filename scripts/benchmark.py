"""Serial warm pipeline latency, with decoded images held in memory; no API I/O."""
import argparse
from dataclasses import asdict
import hashlib
import logging
from pathlib import Path
from time import perf_counter
import numpy as np
from app.api import decode_image
from app.config import Settings
from app.services.analyzer import Analyzer
from scripts.common import environment, save_json


def summarize(times):
    return {'average_ms': float(np.mean(times)), 'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)), 'min_ms': float(min(times)),
            'max_ms': float(max(times)), 'throughput_images_per_second': 1000/float(np.mean(times)),
            'timed_inferences': len(times)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--output', type=Path, default=Path('results/benchmarks/local'))
    args = p.parse_args()
    if args.runs < 1 or args.warmup < 1:
        p.error('runs and warmup must be positive')
    logging.basicConfig(level=logging.INFO)
    try:
        import torch
        torch.set_num_threads(4)
    except ImportError:
        pass
    import cv2
    cv2.setNumThreads(4)
    s = Settings.from_env()
    paths = sorted(p for p in args.images.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))
    if not paths:
        p.error('No benchmark images found')
    images = [decode_image(p.read_bytes(), s) for p in paths]
    report = {'environment': environment(), 'configuration': asdict(s), 'runs': args.runs,
              'warmup': args.warmup, 'number_of_images': len(images), 'threads': 4,
              'timing_scope': 'resize + CV + optional YOLO + fusion + annotation; excludes decode, disk, HTTP and model loading',
              'model': 'YOLOv8 single-class shelf model', 'device': s.yolo_device,
              'images': [{'file': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                          'resolution': [im.shape[1], im.shape[0]]} for p, im in zip(paths, images)], 'modes': {}}
    logging.info('Benchmark environment: %s', report['environment'])
    for mode in ('classical_only', 'hybrid'):
        start = perf_counter()
        analyzer = Analyzer(s, classical_only=mode == 'classical_only')
        load_ms = (perf_counter()-start)*1000
        if mode == 'hybrid' and not analyzer.products.available:
            report['modes'][mode] = {'status': 'unavailable', 'reason': analyzer.products.reason}
            continue
        for i in range(args.warmup):
            analyzer.analyze(images[i % len(images)])
        times = []
        failed = False
        for _ in range(args.runs):
            for image in images:
                start = perf_counter()
                result, *_ = analyzer.analyze(image)
                times.append((perf_counter()-start)*1000)
                if mode == 'hybrid' and result['mode'] != 'hybrid':
                    failed = True
        report['modes'][mode] = ({'status': 'unavailable', 'reason': 'fallback_during_benchmark'} if failed else
            {'status': 'measured', 'cold_construct_ms': load_ms, **summarize(times), 'samples_ms': times})
    save_json(args.output.with_suffix('.json'), report)
    rows = ['# Local warm inference benchmark', '', report['timing_scope'], '',
            f"Images: {len(images)}; repetitions: {args.runs}; warmup: {args.warmup}; device: {s.yolo_device}; threads: 4.", '',
            '| Mode | Mean ms | Median ms | p95 ms | Images/s |', '|---|---:|---:|---:|---:|']
    for mode, r in report['modes'].items():
        rows.append(f"| {mode} | {r['average_ms']:.2f} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['throughput_images_per_second']:.2f} |" if r['status'] == 'measured' else f'| {mode} | unavailable | — | — | — |')
    rows += ['', 'Throughput is serial warm inference throughput, not concurrent API capacity. See JSON for environment, image hashes, original resolutions, configuration and individual timings.']
    args.output.with_suffix('.md').write_text('\n'.join(rows), encoding='utf-8')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
