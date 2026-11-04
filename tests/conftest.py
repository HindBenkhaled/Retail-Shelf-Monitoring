import io
import cv2
import numpy as np
import pytest
from app import create_app
from app.config import Settings
from app.services.analyzer import Analyzer


@pytest.fixture
def settings(tmp_path):
    return Settings(output_dir=str(tmp_path/'results'), yolo_model_path=str(tmp_path/'absent.pt'),
                    min_gap_area=25, min_gap_width=3, min_gap_height=3)


@pytest.fixture
def image():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
    image[35:130, 70:170] = 100
    return image


@pytest.fixture
def client(settings):
    app = create_app(settings)
    app.config['TESTING'] = True
    return app.test_client()


def upload(image, filename='shelf.png'):
    return {'image': (io.BytesIO(cv2.imencode('.png', image)[1].tobytes()), filename)}
