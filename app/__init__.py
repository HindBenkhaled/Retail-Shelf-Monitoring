import logging
from flask import Flask
from app.config import Settings
from app.services.analyzer import Analyzer


def create_app(settings=None, analyzer=None):
    from app.api import api, register_errors
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.config.update(SETTINGS=settings, MAX_CONTENT_LENGTH=settings.max_upload_mb * 1024 * 1024)
    settings.output_path.mkdir(parents=True, exist_ok=True)
    app.extensions['analyzer'] = analyzer if analyzer is not None else Analyzer(settings)
    app.register_blueprint(api)
    register_errors(app)
    logging.getLogger(__name__).info('Shelf monitoring initialized; YOLO available=%s', app.extensions['analyzer'].products.available)
    return app
