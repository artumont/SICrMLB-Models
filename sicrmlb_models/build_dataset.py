import logging
import argparse
import threading
from pathlib import Path
from PIL.Image import Image
from PIL import Image as PILImage

from _types import ImageSize, Variants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets"
RAW_IMAGE_SIZE = ImageSize(width=97, height=120)
NORMALIZED_IMAGE_SIZE = ImageSize(width=120, height=120)
IMAGE_VARIANTS = [
    Variants.GRAYSCALE,
    Variants.CONTRAST,
    Variants.BRIGHT,
    Variants.COMBINED,
    Variants.ORIGINAL,
]


def letterbox_image(image_path: Path, target_size: ImageSize) -> Image:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    tgt_w, tgt_h = int(target_size.width), int(target_size.height)

    with PILImage.open(image_path) as img:
        img_rgb = img.convert("RGB")
        src_w, src_h = img_rgb.size

        # scale to fit target preserving aspect ratio
        scale = min(tgt_w / src_w, tgt_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        img_resized = img_rgb.resize((new_w, new_h), PILImage.Resampling.LANCZOS)

        fill = (114, 114, 114)  # neutral padding color
        canvas = PILImage.new("RGB", (tgt_w, tgt_h), fill)

        # center on canvas
        x = (tgt_w - new_w) // 2
        y = (tgt_h - new_h) // 2
        canvas.paste(img_resized, (x, y))

        return canvas

def normalize_string(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def worker_task(paths: list[Path], worker_id: int, output_dir: Path, variants: list[Variants] = IMAGE_VARIANTS) -> None:
    for img_path in paths:
        try:
            img_out_path = output_dir / "train"
            label_out_path = output_dir / "labels"
            processed_img = letterbox_image(img_path, NORMALIZED_IMAGE_SIZE)
            
            ensure_dir(img_out_path)
            ensure_dir(label_out_path)
            
            if not processed_img:
                logger.warning(f"Worker {worker_id} could not process image: {img_path}")
                continue
            
            # Generate and save variants (use Path / filename)
            if Variants.BRIGHT in variants:
                bright_img = PILImage.eval(processed_img, lambda x: min(255, int(x * 1.2)))
                bright_path = img_out_path / f"{img_path.stem}_bright{img_path.suffix}"
                bright_img.save(bright_path)
            if Variants.GRAYSCALE in variants:
                gray_img = processed_img.convert("L").convert("RGB")
                gray_path = img_out_path / f"{img_path.stem}_grayscale{img_path.suffix}"
                gray_img.save(gray_path)
            if Variants.CONTRAST in variants:
                contrast_img = PILImage.eval(processed_img, lambda x: min(255, max(0, int(128 + 1.5 * (x - 128)))))
                contrast_path = img_out_path / f"{img_path.stem}_contrast{img_path.suffix}"
                contrast_img.save(contrast_path)
            if Variants.COMBINED in variants:
                combined_img = PILImage.eval(processed_img, lambda x: min(255, max(0, int(128 + 1.5 * (x - 128) * 1.2))))
                combined_path = img_out_path / f"{img_path.stem}_combined{img_path.suffix}"
                combined_img.save(combined_path)
            if Variants.ORIGINAL in variants:
                original_path = img_out_path / f"{img_path.stem}_original{img_path.suffix}"
                processed_img.save(original_path)
                
            logger.info(f"Worker {worker_id} processed {img_path.stem} successfully.")
            
            # Write label files (one per generated image variant)
            def write_label(stem: str, suffix: str):
                label_content = f"{normalize_string(img_path.stem)} 0.5 0.5 1.0 1.0"
                label_file_path = label_out_path / f"{stem}{suffix}.txt"
                label_file_path.write_text(label_content + "\n")

            if Variants.ORIGINAL in variants:
                write_label(img_path.stem, "_original")
            if Variants.BRIGHT in variants:
                write_label(img_path.stem, "_bright")
            if Variants.GRAYSCALE in variants:
                write_label(img_path.stem, "_grayscale")
            if Variants.CONTRAST in variants:
                write_label(img_path.stem, "_contrast")
            if Variants.COMBINED in variants:
                write_label(img_path.stem, "_combined")

            logger.info(f"Worker {worker_id} saved label for {img_path.stem} successfully.")
            
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to process {img_path}: {e}")


def build_dataset(
    dataset_name: str,
    raw_data_path: Path,
    output_dir: Path,
    num_workers: int = 4,
) -> None:
    logger.info(f"Building dataset '{dataset_name}' from raw data at {raw_data_path}")

    load = list(raw_data_path.rglob("*.jpg")) + list(raw_data_path.rglob("*.png"))
    balanced_load = balance_load(load, num_workers)

    ensure_dir(output_dir / "train")
    ensure_dir(output_dir / "labels")
    
    worker_threads = []
    for worker_idx, image_paths in enumerate(balanced_load):
        logger.info(f"Worker {worker_idx} processing {len(image_paths)} images")
        thread = threading.Thread(target=worker_task, args=(image_paths, worker_idx+1, output_dir))
        thread.start()
        worker_threads.append(thread)

    for thread in worker_threads:
        logger.debug(f"Waiting for worker thread {thread.name} to finish")
        thread.join()
    
    logger.info(f"Dataset '{dataset_name}' built successfully at {output_dir}")


def balance_load(image_list: list[Path], num_workers: int) -> list[list[Path]]:
    """Distribute images evenly across workers for balanced processing."""
    balanced = [[] for _ in range(num_workers)]
    for idx, image_path in enumerate(image_list):
        worker_idx = idx % num_workers
        balanced[worker_idx].append(image_path)
    return balanced


def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset for SICrMLB models.")
    parser.add_argument(
        "-dn",
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset to build.",
    )
    parser.add_argument(
        "-rd",
        "--raw-data-path",
        type=Path,
        required=True,
        help="Path to the raw data directory.",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=Path,
        default=DATASET_PATH,
        help="Directory to save the processed dataset.",
    )
    parser.add_argument(
        "-nw",
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for data processing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = args.output_dir / args.dataset_name
    
    ensure_dir(output_dir)
    if not args.raw_data_path.exists():
        logger.error(f"Raw data path does not exist: {args.raw_data_path}")
        raise FileNotFoundError(f"Raw data path does not exist: {args.raw_data_path}")

    build_dataset(args.dataset_name, args.raw_data_path, output_dir)
