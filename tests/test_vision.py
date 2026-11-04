from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from app.services.gap_detector import GapDetector, GapResult, filter_regions
from app.services.fusion import fuse, occupancy_mask
from app.services.analyzer import Analyzer
from app.services.product_detector import ProductDetector, adapt_boxes, validate_model


def test_low_texture_and_determinism(settings, image):
    detector = GapDetector(settings)
    result = detector.detect(image)
    assert result.mask.shape == image.shape[:2]
    assert result.mask[80, 120] == 255
    assert result.mask[15, 15] == 0
    assert result.summary()['gap_count'] >= 1
    assert 0 < result.summary()['estimated_empty_area_percentage'] < 60
    assert np.array_equal(result.mask, detector.detect(image).mask)
    assert detector.annotate(image, result).shape == image.shape


def test_exact_area_with_hole(settings):
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 255
    mask[40:60, 40:60] = 0
    clean, regions = filter_regions(mask, settings)
    result = GapResult(clean, regions, {})
    assert result.summary()['gap_area_pixels'] == 6000
    assert result.summary()['estimated_empty_area_percentage'] == 60
    assert regions[0]['area_pixels'] == 6000


def test_filter_small_regions(settings):
    mask = np.zeros((50, 50), np.uint8)
    mask[1:3, 1:3] = 255
    mask[10:30, 10:30] = 255
    clean, regions = filter_regions(mask, settings)
    assert len(regions) == 1
    assert np.count_nonzero(clean) == 400


def test_fusion_splits_and_never_refills(settings):
    mask = np.full((80, 100), 255, np.uint8)
    raw = GapResult(mask, [{'bbox': [0,0,100,80], 'area_pixels':8000}], {})
    detections = [{'bbox':[40,0,60,80]}]
    result = fuse(raw, detections, replace(settings, product_padding=0))
    assert len(result.regions) == 2
    assert np.count_nonzero(result.mask) == 6400
    assert not np.any(result.mask[:, 40:60])
    assert np.all(raw.mask == 255)


def test_occupancy_clips_unions_padding():
    mask = occupancy_mask((10,10), [{'bbox':[-5,-5,5,5]}, {'bbox':[0,0,5,5]}], padding=1)
    assert np.count_nonzero(mask) == 36
    assert not occupancy_mask((10,10), [{'bbox':[20,20,30,30]}]).any()


def test_empty_detections_preserve_raw(settings, image):
    raw = GapDetector(settings).detect(image)
    assert np.array_equal(fuse(raw, [], settings).mask, raw.mask)


class Tensor:
    def __init__(self, values): self.values = np.array(values)
    def cpu(self): return self
    def numpy(self): return self.values


def test_adapter_clips_filters():
    boxes = SimpleNamespace(xyxy=Tensor([[-1,2,110,12],[1,1,4,4],[5,5,4,8]]),
                            conf=Tensor([.9,.1,.9]), cls=Tensor([0,0,0]))
    result = adapt_boxes(SimpleNamespace(boxes=boxes), 100, 10, .25)
    assert len(result) == 1
    assert result[0]['bbox'] == [0,2,100,10]
    assert result[0]['class_name'] == 'product'


def test_adapter_nan_rejected():
    boxes = SimpleNamespace(xyxy=Tensor([[np.nan,0,10,10]]), conf=Tensor([.9]), cls=Tensor([0]))
    with pytest.raises(ValueError): adapt_boxes(SimpleNamespace(boxes=boxes), 20, 20, .25)


def fake_model(error=None):
    model = SimpleNamespace(model=SimpleNamespace(yaml={'yaml_file':'yolov8n.yaml'}), names={0:'product'}, task='detect', ckpt={})
    def predict(*args, **kwargs):
        if error: raise error
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor([[10,10,100,100]]), conf=Tensor([.9]), cls=Tensor([0])))]
    model.predict = predict
    return model


def test_reject_generic_and_wrong_family():
    model = fake_model()
    model.names = {0:'person', 1:'bicycle'}
    with pytest.raises(ValueError): validate_model(model)


def test_finetuned_checkpoint_retains_yolov8_provenance():
    model = fake_model()
    model.model.yaml.pop('yaml_file')
    model.ckpt = {'train_args': {'model': 'yolov8n.pt'}}
    validate_model(model)
    model = fake_model()
    model.model.yaml['yaml_file'] = 'yolo11n.yaml'
    with pytest.raises(ValueError): validate_model(model)


def test_missing_weights_fallback(settings, image):
    analyzer = Analyzer(settings)
    result, annotated, raw, fused = analyzer.analyze(image)
    assert result['mode'] == 'classical_fallback'
    assert result['products_detected'] is None
    assert result['yolo_available'] is False
    assert np.array_equal(raw, fused)


@pytest.mark.parametrize('error', [ImportError('missing'), OSError('bad'), RuntimeError('load'), ValueError('invalid')])
def test_load_failure(settings, tmp_path, error):
    weights = tmp_path/'test.pt'; weights.touch()
    def loader(_): raise error
    detector = ProductDetector(replace(settings, yolo_model_path=str(weights)), loader=loader)
    assert not detector.available
    assert detector.reason == 'model_load_failed'


def test_hybrid_and_inference_fallback(settings, tmp_path, image):
    weights = tmp_path/'test.pt'; weights.touch()
    s = replace(settings, yolo_model_path=str(weights))
    detector = ProductDetector(s, loader=lambda _: fake_model())
    result, *_ = Analyzer(s, product_detector=detector).analyze(image)
    assert result['mode'] == 'hybrid'
    assert result['products_detected'] == 1
    detector.model = fake_model(RuntimeError('recoverable inference error'))
    result, *_ = Analyzer(s, product_detector=detector).analyze(image)
    assert result['mode'] == 'classical_fallback'
    assert detector.reason == 'inference_failed'
    assert detector.predict(image) == []


def test_unexpected_errors_propagate(settings, tmp_path, image):
    weights = tmp_path/'test.pt'; weights.touch()
    detector = ProductDetector(replace(settings, yolo_model_path=str(weights)), loader=lambda _: fake_model(TypeError('programmer bug')))
    with pytest.raises(TypeError): detector.predict(image)


def test_processed_coordinate_space(settings, image):
    result, annotated, *_ = Analyzer(replace(settings, max_image_side=120)).analyze(image)
    assert result['processed_size'] == [120,80]
    assert result['original_size'] == [240,160]
    assert annotated.shape[:2] == (80,120)
