from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path("model") / "rice_model.h5"
DEFAULT_CLASS_INDICES = Path("model") / "class_indices.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_artifacts(model_path: Path, class_indices_path: Path):
    
    # Imported lazily so light-weight consumers can import this module.
    from tensorflow import keras

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train_model.py first."
        )
    if not class_indices_path.is_file():
        raise FileNotFoundError(
            f"Class indices not found: {class_indices_path}. Run train_model.py first."
        )

    model = keras.models.load_model(str(model_path))
    class_indices = json.loads(class_indices_path.read_text(encoding="utf-8"))
    index_to_class = {int(v): k for k, v in class_indices.items()}
    return model, index_to_class


def preprocess_image(image_path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
  
    from PIL import Image, UnidentifiedImageError

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize(target_size, Image.BILINEAR)
            array = np.asarray(img, dtype="float32")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {path}") from exc

    return np.expand_dims(array, axis=0)


def model_input_size(model, fallback: tuple[int, int] = (160, 160)) -> tuple[int, int]:
    """Read the expected (height, width) straight from the loaded model."""
    try:
        _, h, w, _ = model.input_shape
        if h and w:
            return int(h), int(w)
    except (AttributeError, TypeError, ValueError):
        pass
    return fallback


def predict_image(model, index_to_class, image_path, target_size=None, top_k=3):
    """Return (predicted_class, confidence, top-k list) for a single image."""
    if target_size is None:
        target_size = model_input_size(model)
    batch = preprocess_image(image_path, target_size)
    probs = model.predict(batch, verbose=0)[0]
    top_k = max(1, min(top_k, len(probs)))
    order = np.argsort(probs)[::-1][:top_k]
    top = [
        {"variety": index_to_class[int(i)], "probability": float(probs[i])}
        for i in order
    ]
    return top[0]["variety"], top[0]["probability"], top


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect rice grain variety from an image.")
    p.add_argument("--image", help="path to a single image")
    p.add_argument("--image-dir", help="path to a folder of images")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--class-indices", default=str(DEFAULT_CLASS_INDICES))
    p.add_argument("--img-size", type=int, default=0,
                   help="input resolution; 0 = infer from the model")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.image and not args.image_dir:
        raise SystemExit("Provide --image or --image-dir")

    model, index_to_class = load_artifacts(Path(args.model), Path(args.class_indices))
    target_size = (
        model_input_size(model) if not args.img_size else (args.img_size, args.img_size)
    )

    if args.image:
        images = [Path(args.image)]
    else:
        folder = Path(args.image_dir)
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    print(f"Model      : {args.model}")
    print(f"Classes    : {sorted(index_to_class.values())}")
    print("-" * 60)

    for image_path in images:
        variety, confidence, top = predict_image(
            model, index_to_class, image_path, target_size=target_size, top_k=args.top_k
        )
        print(f"{image_path.name}")
        print(f"  Predicted : {variety}  ({confidence * 100:.2f}%)")
        for item in top:
            print(f"    - {item['variety']:<12} {item['probability'] * 100:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
