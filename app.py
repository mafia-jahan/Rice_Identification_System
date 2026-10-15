from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from detect import load_artifacts, predict_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = Path(os.environ.get("RICE_MODEL_PATH", MODEL_DIR / "rice_model.h5"))
CLASS_INDICES_PATH = Path(
    os.environ.get("RICE_CLASS_INDICES", MODEL_DIR / "class_indices.json")
)
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB


MODEL = None
INDEX_TO_CLASS: dict[int, str] = {}
INPUT_SIZE = (160, 160)
LAST_PREDICTION: dict | None = None


def configure_logging(app: Flask) -> None:
    handler = RotatingFileHandler(
        BASE_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)


def load_model_once(app: Flask) -> None:
    
    global MODEL, INDEX_TO_CLASS, INPUT_SIZE
    try:
        MODEL, INDEX_TO_CLASS = load_artifacts(MODEL_PATH, CLASS_INDICES_PATH)
        
        _, h, w, _ = MODEL.input_shape
        INPUT_SIZE = (int(h), int(w))
        app.logger.info(
            "Model loaded: %s | input=%s | classes=%s",
            MODEL_PATH.name,
            INPUT_SIZE,
            sorted(INDEX_TO_CLASS.values()),
        )
    except Exception:  
        app.logger.exception("Failed to load model artifacts")
        raise


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_upload(file_storage) -> str | None:
   
    if file_storage is None or file_storage.filename == "":
        return "Please choose an image file to upload."
    if not allowed_file(file_storage.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return f"Unsupported file type. Allowed types: {allowed}."
    return None


def save_as_jpg(image_source, dest: Path) -> None:
    
    import io

    from PIL import Image, UnidentifiedImageError

    stream = image_source.stream if hasattr(image_source, "stream") else io.BytesIO(image_source)
    try:
        with Image.open(stream) as img:
            img.convert("RGB").save(dest, format="JPEG", quality=92)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Not a valid image file.") from exc


def render_index(**context):
    return render_template("index.html", classes=sorted(INDEX_TO_CLASS.values()), **context)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(32))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging(app)
    load_model_once(app)

    @app.route("/")
    def index():
        return render_index()

    @app.route("/detect", methods=["POST"])
    def detect():
        global LAST_PREDICTION

        file = request.files.get("image")
        error = validate_upload(file)
        if error:
            app.logger.warning("Rejected upload: %s", error)
            return render_index(error=error), 400

       
        safe_name = f"{uuid.uuid4().hex}.jpg"
        saved_path = UPLOAD_DIR / safe_name
        try:
            save_as_jpg(file, saved_path)
        except ValueError:
            app.logger.warning("Unreadable image upload: %s", file.filename)
            return render_index(error="That file is not a valid image."), 400
        except Exception:  
            app.logger.exception("Could not save upload")
            return render_index(error="Could not save the image."), 500

        try:
            variety, confidence, top_k = predict_image(
                MODEL, INDEX_TO_CLASS, saved_path, target_size=INPUT_SIZE, top_k=5
            )
        except Exception:  
            app.logger.exception("Prediction failed for %s", safe_name)
            saved_path.unlink(missing_ok=True)
            return render_index(
                error="Could not process this image. Try a different file."
            ), 400

        LAST_PREDICTION = {
            "variety": variety,
            "confidence": round(confidence * 100, 2),
            "top_k": [
                {"variety": item["variety"], "confidence": round(item["probability"] * 100, 2)}
                for item in top_k
            ],
            "image_url": url_for("static", filename=f"uploads/{safe_name}"),
            "filename": file.filename,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        app.logger.info(
            "Prediction: %s (%.2f%%) for %s", variety, confidence * 100, file.filename
        )
        return render_template("result.html", result=LAST_PREDICTION, classes=sorted(INDEX_TO_CLASS.values()))

    @app.route("/result")
    def result():
        if LAST_PREDICTION is None:
            return redirect(url_for("index"))
        return render_template("result.html", result=LAST_PREDICTION, classes=sorted(INDEX_TO_CLASS.values()))

    @app.route("/predict", methods=["POST"])
    def predict_api():
        """JSON API. Accepts multipart 'image' file or JSON {"image_base64": ...}."""
        import base64
        import binascii

        tmp_path: Path | None = None
        try:
            if "image" in request.files:
                file = request.files["image"]
                if not allowed_file(file.filename or ""):
                    return jsonify({"error": "Unsupported file type."}), 400
                tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
                try:
                    save_as_jpg(file, tmp_path)
                except ValueError:
                    return jsonify({"error": "Not a valid image file."}), 400
            elif request.is_json and request.json.get("image_base64"):
                try:
                    raw = base64.b64decode(request.json["image_base64"], validate=True)
                except (binascii.Error, ValueError):
                    return jsonify({"error": "Invalid base64 image."}), 400
                tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
                try:
                    save_as_jpg(raw, tmp_path)
                except ValueError:
                    return jsonify({"error": "Not a valid image file."}), 400
            else:
                return jsonify({"error": "No image provided."}), 400

            variety, confidence, top_k = predict_image(
                MODEL, INDEX_TO_CLASS, tmp_path, target_size=INPUT_SIZE, top_k=5
            )
            return jsonify(
                {
                    "prediction": variety,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        item["variety"]: round(item["probability"], 4) for item in top_k
                    },
                }
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_loaded": MODEL is not None,
                "classes": sorted(INDEX_TO_CLASS.values()),
            }
        )

    @app.errorhandler(404)
    def not_found(_error):
        return render_index(error="Page not found."), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_index(error="File is too large. Maximum size is 10 MB."), 413

    return app


app = create_app()


if __name__ == '__main__':
    app.run(debug=True,port=5000)