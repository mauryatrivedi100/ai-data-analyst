"""Flask application entry point for the AI Data Analyst backend."""

import math
import os

from flask import Flask
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from routes import bp as routes_bp


class NaNSafeJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that converts NaN/Infinity to None for safe serialization."""

    def default(self, o):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return super().default(o)


def create_app():
    """Application factory for the Flask backend."""
    app = Flask(__name__)
    app.json_provider_class = NaNSafeJSONProvider
    app.json = NaNSafeJSONProvider(app)

    # Configuration
    app.config["UPLOAD_FOLDER"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "uploads"
    )
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # CORS — allow requests from the React frontend dev server
    CORS(app, origins=["http://localhost:5173"])

    # Register route blueprints
    app.register_blueprint(routes_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
