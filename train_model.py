from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "model"
MODEL_FILENAME = "rice_model.h5"
CLASS_INDICES_FILENAME = "class_indices.json"
METRICS_FILENAME = "metrics_report.txt"
AUTOTUNE = tf.data.AUTOTUNE


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_datasets(data_dir: Path, img_size: int, batch_size: int, seed: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{val_dir}'. Run split_dataset.py first."
        )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    if class_names != val_ds.class_names:
        raise RuntimeError("train/ and val/ class folders do not match.")

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes: int, img_size: int) -> keras.Model:
    
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = data_augmentation(inputs)
    # MobileNetV2 preprocessing: scale [0, 255] -> [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="rice_variety_classifier")
    return model, base


def make_callbacks(model_path: Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]


def evaluate(model: keras.Model, val_ds: tf.data.Dataset, class_names: list[str]):
   
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in val_ds])
    probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    loss, acc = model.evaluate(val_ds, verbose=0)
    return y_true, y_pred, float(loss), float(acc)


def write_metrics_report(
    path: Path,
    *,
    class_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_loss: float,
    val_acc: float,
    history: dict,
    train_count: int,
    val_count: int,
    config: dict,
    duration_s: float,
) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("RICE VARIETY CLASSIFIER - EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Training time    : {duration_s:.1f} s ({duration_s / 60:.1f} min)")
    lines.append("")
    lines.append("-- Dataset ------------------------------------------------------------")
    lines.append(f"Classes          : {len(class_names)} ({', '.join(class_names)})")
    lines.append(f"Train images     : {train_count}")
    lines.append(f"Validation images: {val_count}")
    lines.append("")
    lines.append("-- Configuration ------------------------------------------------------")
    for key, value in config.items():
        lines.append(f"{key:<17}: {value}")
    lines.append("")
    lines.append("-- Overall metrics ----------------------------------------------------")
    lines.append(f"Validation loss  : {val_loss:.4f}")
    lines.append(f"Validation acc   : {val_acc:.4f} ({val_acc * 100:.2f}%)")
    lines.append("")
    lines.append("-- Per-epoch history --------------------------------------------------")
    lines.append(
        f"{'epoch':>5} {'loss':>9} {'acc':>9} {'val_loss':>10} {'val_acc':>9} {'lr':>10}"
    )
    hist = history or {}
    losses = hist.get("loss", [])
    accs = hist.get("accuracy", [])
    vlosses = hist.get("val_loss", [])
    vaccs = hist.get("val_accuracy", [])
    lrs = hist.get("learning_rate", hist.get("lr", []))
    for i in range(len(losses)):
        lines.append(
            f"{i + 1:>5} "
            f"{losses[i]:>9.4f} {accs[i]:>9.4f} "
            f"{vlosses[i]:>10.4f} {vaccs[i]:>9.4f} "
            f"{(lrs[i] if i < len(lrs) else float('nan')):>10.2e}"
        )
    lines.append("")
    lines.append("-- Classification report (validation set) -----------------------------")
    lines.append(
        classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    )
    lines.append("-- Confusion matrix (rows = true, cols = predicted) -------------------")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    header = " " * 14 + "".join(f"{name[:10]:>12}" for name in class_names)
    lines.append(header)
    for i, name in enumerate(class_names):
        row = "".join(f"{int(v):>12}" for v in cm[i])
        lines.append(f"{name[:12]:<12}  {row}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of report")
    lines.append("=" * 70)
    path.write_text("\n".join(lines), encoding="utf-8")


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sum(
        1
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the rice variety classifier.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6, help="epochs for the frozen-head stage")
    p.add_argument("--fine-tune", action="store_true", help="unfreeze the top of MobileNetV2")
    p.add_argument("--fine-tune-epochs", type=int, default=4)
    p.add_argument("--fine-tune-layers", type=int, default=30, help="last N base layers to unfreeze")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seeds(args.seed)

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    class_indices_path = model_dir / CLASS_INDICES_FILENAME
    metrics_path = model_dir / METRICS_FILENAME

    print("=" * 60)
    print("Rice variety classifier - training")
    print("=" * 60)

    train_ds, val_ds, class_names = build_datasets(
        data_dir, args.img_size, args.batch_size, args.seed
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model, base = build_model(num_classes, args.img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    started = time.time()
    callbacks = make_callbacks(model_path)

    print("\n[Stage 1] Training classification head (backbone frozen)")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    full_history = {k: list(v) for k, v in hist1.history.items()}

    if args.fine_tune:
        print(f"\n[Stage 2] Fine-tuning last {args.fine_tune_layers} backbone layers")
        base.trainable = True
        for layer in base.layers[: max(0, len(base.layers) - args.fine_tune_layers)]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate / 20),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        for k, v in hist2.history.items():
            full_history.setdefault(k, []).extend(v)

    duration = time.time() - started

    
    if not model_path.exists():
        model.save(str(model_path))

    best_model = keras.models.load_model(str(model_path))
    y_true, y_pred, val_loss, val_acc = evaluate(best_model, val_ds, class_names)

   
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    class_indices_path.write_text(
        json.dumps(class_indices, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config = {
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_head": args.epochs,
        "fine_tune": args.fine_tune,
        "fine_tune_epochs": args.fine_tune_epochs if args.fine_tune else 0,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    write_metrics_report(
        metrics_path,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
        val_loss=val_loss,
        val_acc=val_acc,
        history=full_history,
        train_count=count_images(data_dir / "train"),
        val_count=count_images(data_dir / "val"),
        config=config,
        duration_s=duration,
    )

    print("\n" + "=" * 60)
    print(f"Best validation accuracy : {val_acc * 100:.2f}%")
    print(f"Model saved to           : {model_path}")
    print(f"Class indices saved to   : {class_indices_path}")
    print(f"Metrics report saved to  : {metrics_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
