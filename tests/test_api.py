import io
import os
import time
from dataclasses import replace
import pytest
from app import create_app
from app.api import decode_image
from tests.conftest import upload


def test_home_and_health(client):
    assert client.get('/').status_code == 200
    assert b'Analyze a shelf' in client.get('/').data
    data = client.get('/health').get_json()
    assert data['status'] == 'ok' and data['mode'] == 'classical_fallback'


def test_valid_upload_result_and_unique_id(client, image):
    response = client.post('/api/analyze', data=upload(image))
    assert response.status_code == 200
    data = response.get_json()
    assert {'analysis_id','mode','raw_classical','fused','gaps','processing_time_ms','estimated_empty_area_percentage'} <= data.keys()
    result = client.get(data['annotated_image_url'])
    assert result.status_code == 200 and result.mimetype == 'image/jpeg'
    assert result.data[:2] == b'\xff\xd8'
    assert client.post('/api/analyze', data=upload(image)).json['analysis_id'] != data['analysis_id']


@pytest.mark.parametrize('name,content,status', [('bad.txt',b'bad',415),('fake.png',b'not an image',400),('fake.jpg',b'',400)])
def test_invalid_file(client, name, content, status):
    assert client.post('/api/analyze', data={'image':(io.BytesIO(content),name)}).status_code == status


def test_missing_upload(client):
    assert client.post('/api/analyze').status_code == 400


def test_oversized_upload(settings):
    client = create_app(replace(settings, max_upload_mb=1)).test_client()
    response = client.post('/api/analyze', data={'image':(io.BytesIO(b'x'*(1024*1024+1)),'large.png')})
    assert response.status_code == 413
    assert 'error' in response.json


def test_pixel_limit(settings, image):
    client = create_app(replace(settings, max_image_pixels=100)).test_client()
    assert client.post('/api/analyze', data=upload(image)).status_code == 400


@pytest.mark.parametrize('path', ['/results/not-an-id.jpg','/results/'+'a'*32+'.jpg','/results/../../.env','/missing'])
def test_safe_result_paths(client, path):
    assert client.get(path).status_code == 404


def test_expiration_and_result_limit(settings, image):
    app = create_app(replace(settings, max_results=1, result_ttl_seconds=1))
    client = app.test_client()
    first = client.post('/api/analyze', data=upload(image)).json
    second = client.post('/api/analyze', data=upload(image)).json
    assert client.get(first['annotated_image_url']).status_code == 404
    file = settings.output_path/(second['analysis_id']+'.jpg')
    os.utime(file, (time.time()-5,time.time()-5))
    assert client.get(second['annotated_image_url']).status_code == 404


def test_unexpected_error_is_logged_and_sanitized(client, image, monkeypatch, caplog):
    def fail(_): raise TypeError('secret local path C:/private/file')
    monkeypatch.setattr(client.application.extensions['analyzer'], 'analyze', fail)
    response = client.post('/api/analyze', data=upload(image))
    assert response.status_code == 500
    assert 'private' not in response.get_data(as_text=True)
    assert 'Unexpected processing error' in caplog.text
