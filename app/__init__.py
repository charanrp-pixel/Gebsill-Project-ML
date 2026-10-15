"""Flask application factory for Student Performance Prediction System."""

from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["JSON_SORT_KEYS"] = False

    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
