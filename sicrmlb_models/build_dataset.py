from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import argparse
import csv
import json
import sys

TARGET_SIZE = (224, 224)
VARIANTS = ("original", "grayscale", "bright", "combo")
DEFAULT_BRIGHT_FACTOR = 1.2

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def process_image(img: Image.Image, variant: str, size=TARGET_SIZE, bright_factor=DEFAULT_BRIGHT_FACTOR) -> Image.Image:
    # Work on a copy
    if variant == "original":
        out = img.convert("RGBA").convert("RGB")
    elif variant == "grayscale":
        out = ImageOps.grayscale(img).convert("RGB")
    elif variant == "bright":
        out = img.convert("RGB")
        out = ImageEnhance.Brightness(out).enhance(bright_factor)
    elif variant == "combo":
        # Combine transformations: grayscale -> brighten -> contrast -> slight blur
        out = ImageOps.grayscale(img)
        out = ImageEnhance.Brightness(out).enhance(bright_factor * 1.05)
        out = ImageEnhance.Contrast(out).enhance(1.05)
        out = out.filter(ImageFilter.GaussianBlur(radius=0.5))
        out = out.convert("RGB")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Resize with anti-aliasing
    out = out.resize(size, Image.Resampling.LANCZOS)
    return out

def iter_images(raw_dir: Path, patterns=("*.png", "*.jpg", "*.jpeg")):
    for pat in patterns:
        for p in sorted(raw_dir.glob(pat)):
            if p.is_file():
                yield p

def build_dataset(raw_dir: Path, out_root: Path, size=TARGET_SIZE, bright_factor=DEFAULT_BRIGHT_FACTOR, out_format="png", write_manifest=True):
    raw_dir = Path(raw_dir)
    out_root = Path(out_root)
    ensure_dir(out_root)

    manifest_rows = []
    # Variables for mean/std computation
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sumsq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    images_processed = 0

    for src in iter_images(raw_dir):
        name = src.stem
        card_out_dir = out_root / name
        ensure_dir(card_out_dir)

        try:
            with Image.open(src) as img:
                img = img.convert("RGBA")
                for variant in VARIANTS:
                    out_img = process_image(img, variant, size=size, bright_factor=bright_factor)
                    out_name = f"{variant}.{out_format}"
                    out_path = card_out_dir / out_name
                    out_img.save(out_path, format=out_format.upper())

                    # Update manifest
                    manifest_rows.append({
                        "path": str(out_path.relative_to(out_root.parent)),  # relative to project datasets folder
                        "label": name,
                        "variant": variant,
                        "width": out_img.width,
                        "height": out_img.height
                    })

                    # Update stats
                    arr = np.asarray(out_img).astype(np.float32) / 255.0
                    if arr.ndim == 2:  # grayscale
                        arr = np.stack([arr]*3, axis=-1)
                    h, w, c = arr.shape
                    pixels = h * w
                    total_pixels += pixels
                    channel_sum += arr.reshape(-1, 3).sum(axis=0)
                    channel_sumsq += (arr.reshape(-1, 3) ** 2).sum(axis=0)

                images_processed += 1
        except Exception as e:
            print(f"Warning: failed to process {src}: {e}", file=sys.stderr)
            continue

    # Write manifest CSV
    manifest_path = out_root / "manifest.csv"
    if write_manifest:
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "label", "variant", "width", "height"])
            writer.writeheader()
            for r in manifest_rows:
                writer.writerow(r)

    # Compute mean/std
    if total_pixels > 0:
        mean = (channel_sum / total_pixels).tolist()
        var = (channel_sumsq / total_pixels) - np.array(mean) ** 2
        std = np.sqrt(var).tolist()
    else:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]

    stats = {"mean": mean, "std": std, "images": images_processed, "variants_per_image": len(VARIANTS)}
    stats_path = out_root / "stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"Processed {images_processed} source images -> variants saved under {out_root}")
    print(f"Stats saved to {stats_path}")
    print(f"Manifest saved to {manifest_path}")

    return {"manifest": str(manifest_path), "stats": str(stats_path), "processed": images_processed}

def parse_args():
    p = argparse.ArgumentParser(description="Build image dataset with variants for model training.")
    p.add_argument("--raw", "-r", type=str, default=str(Path(__file__).parent / "raw_data" / "cards"), help="Raw images directory (cards)")
    p.add_argument("--out", "-o", type=str, default=str(Path(__file__).parent / "datasets" / "cards"), help="Output dataset root")
    p.add_argument("--size", "-s", type=int, nargs=2, default=TARGET_SIZE, help="Target size W H (default 224 224)")
    p.add_argument("--bright", type=float, default=DEFAULT_BRIGHT_FACTOR, help="Brightness factor for bright/combo variants (default 1.2)")
    p.add_argument("--format", "-f", choices=["png", "jpg"], default="png", help="Output image format (png or jpg)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.out)

    if not raw.exists():
        print(f"Raw directory not found: {raw}", file=sys.stderr)
        sys.exit(2)

    build_dataset(raw, out, size=tuple(args.size), bright_factor=args.bright, out_format=args.format)
