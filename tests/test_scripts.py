from dataclasses import replace
import pytest
from app.config import Settings
from scripts.prepare_dataset import yolo_box
from scripts.benchmark import summarize


def test_dataset_xywh_conversion_and_clipping():
    assert yolo_box([10,20,20,40],100,100) == '0 0.20000000 0.40000000 0.20000000 0.40000000'
    assert yolo_box([-10,0,20,10],100,100) == '0 0.05000000 0.05000000 0.10000000 0.10000000'
    with pytest.raises(ValueError): yolo_box([100,100,5,5],100,100)


def test_benchmark_statistics():
    r = summarize([10,20,30,40])
    assert r['average_ms'] == 25 and r['median_ms'] == 25
    assert r['p95_ms'] == pytest.approx(38.5)
    assert r['throughput_images_per_second'] == 40


@pytest.mark.parametrize('values', [{'morph_kernel':2},{'texture_window':0},{'yolo_confidence':2},{'max_image_side':0},{'min_gap_area':-1}])
def test_invalid_settings(values):
    with pytest.raises(ValueError): Settings(**values)


def test_environment_config(monkeypatch):
    monkeypatch.setenv('MIN_GAP_AREA','123')
    monkeypatch.setenv('LAPLACIAN_THRESHOLD','9.5')
    s = Settings.from_env()
    assert s.min_gap_area == 123 and s.laplacian_threshold == 9.5
