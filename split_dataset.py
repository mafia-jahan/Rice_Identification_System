from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> list[Path]:
    """Return every image file directly inside folder."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def reset_dir(path: Path) -> None:
    """Delete path if it exists and recreate it empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path, mode: str) -> None:
    """Copy or symlink src to dst."""
    if dst.exists():
        dst.unlink()
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            print(
                f"  ! symlink failed for {src.name}; falling back to copy "
                "(enable Developer Mode or run as admin for symlinks)",
                file=sys.stderr,
            )
    shutil.copy2(src, dst)


def split_dataset(
    source: Path,
    dest: Path,
    val_ratio: float,
    seed: int,
    train_per_class: int | None,
    val_per_class: int | None,
    mode: str,
) -> dict[str, dict[str, int]]:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    class_dirs = sorted(d for d in source.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found inside {source}")

    train_root = dest / "train"
    val_root = dest / "val"
    reset_dir(train_root)
    reset_dir(val_root)

    rng = random.Random(seed)
    summary: dict[str, dict[str, int]] = {}

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            print(f"  ! skipping empty class: {class_dir.name}")
            continue

        rng.shuffle(images)

        if train_per_class is not None or val_per_class is not None:
            # Explicit counts requested (useful for fast experiments).
            n_train = train_per_class if train_per_class is not None else len(images)
            n_val = val_per_class if val_per_class is not None else int(len(images) * val_ratio)
            train_files = images[:n_train]
            val_files = images[n_train : n_train + n_val]
        else:
            # Ratio based split over the whole class.
            n_val = max(1, int(round(len(images) * val_ratio)))
            val_files = images[:n_val]
            train_files = images[n_val:]

        for split_name, files in (("train", train_files), ("val", val_files)):
            out_dir = dest / split_name / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(files):
                transfer(src, out_dir / src.name, mode)
                if (i + 1) % 2000 == 0:
                    print(f"    {split_name}/{class_dir.name}: {i + 1}/{len(files)}")

        summary[class_dir.name] = {"train": len(train_files), "val": len(val_files)}
        print(
            f"  {class_dir.name:<12} train={len(train_files):<6} val={len(val_files):<6}"
        )

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split rice dataset into train/val folders.")
    parser.add_argument("--source", default="Rice_Image_Dataset", help="raw dataset root")
    parser.add_argument("--dest", default="data", help="output root (train/ and val/)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed for a reproducible split")
    parser.add_argument("--train-per-class", type=int, default=None, help="cap training images per class")
    parser.add_argument("--val-per-class", type=int, default=None, help="cap validation images per class")
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="how to place images into data/ (default: copy)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = Path(args.source)
    dest = Path(args.dest)

    print(f"Source : {source.resolve()}")
    print(f"Dest   : {dest.resolve()}")
    print(f"Mode   : {args.mode}   val_ratio={args.val_ratio}   seed={args.seed}")
    if args.train_per_class or args.val_per_class:
        print(
            f"Limits : train_per_class={args.train_per_class} "
            f"val_per_class={args.val_per_class}"
        )
    print("-" * 60)

    summary = split_dataset(
        source=source,
        dest=dest,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        mode=args.mode,
    )

    total_train = sum(v["train"] for v in summary.values())
    total_val = sum(v["val"] for v in summary.values())
    print("-" * 60)
    print(f"Classes      : {len(summary)} -> {', '.join(summary)}")
    print(f"Train images : {total_train}")
    print(f"Val images   : {total_val}")
    print(f"Done. Data ready in '{dest}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
