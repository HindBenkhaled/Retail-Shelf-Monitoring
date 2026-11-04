import importlib.metadata
import platform
import json
from pathlib import Path


def environment():
    versions = {}
    for name in ('torch', 'ultralytics', 'opencv-python', 'numpy', 'Flask', 'Pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'packages': versions}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
