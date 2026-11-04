import io
import logging
import re
import time
import uuid
from threading import Lock
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import HTTPException

api = Blueprint('api', __name__)
log = logging.getLogger(__name__)
output_lock = Lock()


def decode_image(data, settings):
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Only JPEG, PNG and WebP images are supported')
            if source.width * source.height > settings.max_image_pixels:
                raise ValueError('Image dimensions exceed the configured pixel limit')
            source.load()
            rgb = np.array(ImageOps.exif_transpose(source).convert('RGB'))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('The uploaded file could not be decoded as an image') from exc


@api.get('/')
def index():
    return render_template('index.html', max_mb=current_app.config['SETTINGS'].max_upload_mb)


@api.get('/health')
def health():
    detector = current_app.extensions['analyzer'].products
    return jsonify(status='ok', yolo_available=detector.available,
                   mode='hybrid' if detector.available else 'classical_fallback',
                   fallback_reason=detector.reason)


@api.post('/api/analyze')
def analyze():
    upload = request.files.get('image')
    if upload is None or not upload.filename:
        return jsonify(error='Upload an image using the multipart field "image"'), 400
    if upload.filename.rsplit('.', 1)[-1].lower() not in ('jpg', 'jpeg', 'png', 'webp'):
        return jsonify(error='Unsupported file extension; use JPEG, PNG or WebP'), 415
    settings = current_app.config['SETTINGS']
    try:
        image = decode_image(upload.read(), settings)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    result, annotated, _, _ = current_app.extensions['analyzer'].analyze(image)
    analysis_id = uuid.uuid4().hex
    path = settings.output_path
    with output_lock:
        files = sorted(path.glob('*.jpg'), key=lambda p: p.stat().st_mtime)
        for file in files:
            if time.time() - file.stat().st_mtime > settings.result_ttl_seconds:
                file.unlink(missing_ok=True)
        files = sorted(path.glob('*.jpg'), key=lambda p: p.stat().st_mtime)
        for file in files[:max(0, len(files)-settings.max_results+1)]:
            file.unlink(missing_ok=True)
        if not cv2.imwrite(str(path / f'{analysis_id}.jpg'), annotated):
            raise OSError('Failed to save annotation')
    result.update(analysis_id=analysis_id, annotated_image_url=url_for('api.result_image', analysis_id=analysis_id))
    return jsonify(result)


@api.get('/results/<analysis_id>.jpg')
def result_image(analysis_id):
    if not re.fullmatch('[a-f0-9]{32}', analysis_id):
        return jsonify(error='Result not found'), 404
    settings = current_app.config['SETTINGS']
    path = settings.output_path / f'{analysis_id}.jpg'
    with output_lock:
        if not path.is_file() or time.time()-path.stat().st_mtime > settings.result_ttl_seconds:
            return jsonify(error='Result not found or expired'), 404
        return send_from_directory(settings.output_path, path.name, mimetype='image/jpeg', max_age=0)


def register_errors(app):
    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error='Upload exceeds the configured size limit'), 413

    @app.errorhandler(Exception)
    def handle_error(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.name), error.code
        log.exception('Unexpected processing error')
        return jsonify(error='Analysis failed; see server logs for details'), 500
